from dataclasses import asdict
import json
from pathlib import Path
import queue
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from nervus import Code, Finish, ModelContext, ScriptedModel, Session
from nervus.host.history import ContextBudgetError, History, model_context
from nervus.host.terminal import Job
from nervus.models.deepseek import DeepSeekFlash


def options(**changes):
    return SimpleNamespace(**dict(debug=False, context_bytes=98304, max_decisions=8,
                                  call_budget=20, stop_grace=2, **changes))


class HistoryTests(unittest.TestCase):
    def test_overflow_is_retained_and_blocks_next_input_until_explicit_clear(self):
        history = History(150)
        self.assertEqual(history.begin('remember'), [])
        self.assertTrue(history.finish('x'*200))
        snapshot = list(history.messages)
        with self.assertRaises(ContextBudgetError):
            history.begin('next')
        self.assertEqual(history.messages, snapshot)
        self.assertEqual(history.messages[-1]['content'], 'x'*200)
        history.clear()
        self.assertTrue(history.cleared)
        self.assertEqual(history.begin('next'), [])

    def test_context_cap_refuses_without_truncating_feedback(self):
        context = ModelContext('task', 1, 1, 1, 20, (), (), ())
        with self.assertRaises(ContextBudgetError):
            model_context(context, [], '/project', 50)
        self.assertEqual(context.input, 'task')

    def test_history_reaches_adapter_actual_request(self):
        class Response:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, size):
                return json.dumps({'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': '{"type":"finish","answer":"ok"}'}}]}).encode()
        seen = []
        def send(req, **kwargs):
            seen.append(json.loads(req.data))
            return Response()
        context = ModelContext('continue', 2, 1, 1, 20, (), (), ())
        history = [{'role': 'user', 'content': 'save data'}, {'role': 'assistant', 'content': 'saved'}]
        enriched = model_context(context, history, '/project', 98304)
        with patch('urllib.request.urlopen', side_effect=send):
            self.assertEqual(DeepSeekFlash(api_key='offline-test').decide(enriched), Finish('ok'))
        envelope = json.loads(json.loads(seen[0]['messages'][1]['content'])['input'])
        self.assertEqual(envelope['conversation'], history)
        self.assertEqual(envelope['task'], 'continue')
        self.assertEqual(envelope['working_directory'], '/project')

    def test_separate_history_and_state_across_jobs_and_clear(self):
        history = History(10000)
        events = queue.Queue()
        with Session() as session, patch('urllib.request.urlopen', side_effect=AssertionError('Network forbidden')):
            first = Job(session, ScriptedModel([Code('saved = 42'), Finish('saved')]), 'remember',
                        history.begin('remember'), False, Path.cwd(), events, options())
            first.run()
            self.assertIsNone(first.error)
            history.finish('saved')
            model = ScriptedModel([Code('assert saved == 42'), Finish('reused')])
            second = Job(session, model, 'continue', history.begin('continue'), False, Path.cwd(), events, options())
            second.run()
            self.assertEqual(json.loads(model.requests[0].input)['conversation'], [
                {'role': 'user', 'content': 'remember'}, {'role': 'assistant', 'content': 'saved'}])
            history.clear()
            third_model = ScriptedModel([Code('assert saved == 42'), Finish('still saved')])
            third = Job(session, third_model, 'check', history.begin('check'), True, Path.cwd(), events, options())
            third.run()
            self.assertEqual(json.loads(third_model.requests[0].input)['conversation'], [])
            self.assertTrue(json.loads(third_model.requests[0].input)['history_cleared_by_user'])
            self.assertIsNone(third.error)

    def test_context_overflow_never_calls_model_and_drains_turn(self):
        with Session() as session:
            model = ScriptedModel([Finish('must not run')])
            args = options()
            args.context_bytes = 20
            job = Job(session, model, 'input', [], False, Path.cwd(), queue.Queue(), args)
            job.run()
            self.assertIsInstance(job.error, ContextBudgetError)
            self.assertEqual(model.requests, [])
            session.begin_turn()
            session.end_turn()

    def test_debug_output_deltas_keep_appends_without_repeating_chunks(self):
        events = queue.Queue()
        args = options()
        args.debug = True
        job = Job(None, None, '', [], False, Path.cwd(), events, args)
        for text in ('a', 'ab', 'ab'):
            job.observe([], {'chunks': [{'execution': 1, 'channel': 'stdout', 'text': text}], 'truncated': False})
        values = []
        while not events.empty():
            values.append(events.get()[2][1]['chunks'][0]['text'])
        self.assertEqual(values, ['a', 'b'])

    def test_stopped_late_model_does_not_emit_code_or_affect_new_turn(self):
        entered, release = threading.Event(), threading.Event()
        events = queue.Queue()
        def blocked(context):
            entered.set()
            release.wait(5)
            return Code('stale = True')
        with Session() as session:
            job = Job(session, ScriptedModel([blocked]), 'task', [], False, Path.cwd(), events, options())
            job.thread.start()
            try:
                self.assertTrue(entered.wait(3))
                self.assertTrue(job.stop()['stopped'])
                session.begin_turn()
                session.execute('saved = 42')
                release.set()
                job.thread.join(3)
                session.execute('assert saved == 42 and "stale" not in globals()')
                session.end_turn()
            finally:
                release.set()
            emitted = []
            while not events.empty():
                emitted.append(events.get())
            self.assertFalse(any(kind == 'notice' and '执行 Python' in value for _, kind, value in emitted))
