from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from nervus import Finish, ModelContext, Session
from nervus.host.capability_experiment import Experiment, QUESTION
from nervus.host.history import model_context
from nervus.host.terminal import process_capability
from nervus.models.deepseek import DeepSeekFlash
from tests.unit.test_deepseek import response
from tests.host.test_terminal import Terminal


class CapabilityExperimentTests(unittest.TestCase):
    def test_terminal_records_actual_434_and_manually_checks_answer_433(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / 'trace.jsonl'
            answers = ['run、echo、add、upper，共 4 个。', 'run、echo、add，共 3 个。', 'run、echo、add，共 3 个。']
            terminal = Terminal(root, [[{'type': 'finish', 'answer': a}] for a in answers], '--cap-log', str(log))
            try:
                for number, (count, answer) in enumerate(zip((4, 3, 4), answers), 1):
                    terminal.send(f'/cap-env {count}')
                    terminal.expect('Host 已发布：')
                    terminal.expect('你> ')
                    terminal.send('/cap-ask')
                    terminal.expect(answer)
                    terminal.expect('你> ')
                    terminal.send('/cap-check ' + ('4 run echo add upper' if number == 1 else '3 run echo add'))
                    terminal.expect('"count_match": ' + ('false' if number == 3 else 'true'))
                    terminal.expect('你> ')
                terminal.send('/cap-ask')
                terminal.expect('三轮已完成')
                terminal.expect('你> ')
            finally:
                self.assertEqual(terminal.close(), 0)
            events = [json.loads(line) for line in log.read_text().splitlines()]
            publications = [e for e in events if e['event'] == 'publication']
            self.assertEqual([len(e['capabilities']) for e in publications], [4, 3, 4])
            self.assertEqual(publications[0]['capabilities'], publications[2]['capabilities'])
            contexts = [e for e in events if e['event'] == 'model_context']
            self.assertEqual([len(e['context']['capabilities']) for e in contexts], [4, 3, 4])
            self.assertEqual([e['context']['turn'] for e in contexts], [1, 2, 3])
            self.assertEqual([e['context']['revision'] for e in contexts], [2, 3, 4])
            for index, event in enumerate(contexts):
                input = json.loads(event['context']['input'])
                self.assertEqual(input['task'], QUESTION)
                self.assertEqual(len(input['conversation']), 2 * index)
                self.assertNotIn('/cap-', event['context']['input'])
                self.assertNotIn('Host 已发布', event['context']['input'])
                self.assertEqual([m['content'] for m in input['conversation'] if m['role'] == 'user'], [QUESTION]*index)
            self.assertIsNone(contexts[1]['context']['capability_changes'][0]['after'])
            self.assertIsNone(contexts[2]['context']['capability_changes'][0]['before'])
            completed = [e for e in events if e['event'] == 'trial_finished']
            self.assertEqual([e['raw_answer'] for e in completed], answers)
            self.assertTrue(all(e['inspection']['code_inspection'] == 'no_code' for e in completed))
            checks = [e for e in events if e['event'] == 'answer_check']
            self.assertFalse(checks[-1]['names_match'])
            self.assertFalse(checks[-1]['count_match'])
            self.assertNotIn('offline-test', log.read_text())

    def test_restore_alias_identity_and_actual_inspection_without_extra_capability(self):
        with tempfile.TemporaryDirectory() as directory, Session() as session:
            experiment = Experiment(Path(directory)/'trace.jsonl', session, process_capability(), {})
            try:
                experiment.publish(4)
                session.begin_turn()
                session.execute('alias = tools.upper\nif False: inspect_workspace()')
                session.end_turn()
                self.assertEqual(session.inspections, ())
                experiment.publish(3)
                session.begin_turn()
                from nervus import CodeExecutionError
                with self.assertRaises(CodeExecutionError):
                    session.execute("await alias('x')")
                session.end_turn()
                experiment.publish(4)
                session.begin_turn()
                result = session.execute("catalog = inspect_workspace()\nvalue = await alias('x')", exports=['value'])
                self.assertEqual(result, {'value': 'X'})
                self.assertNotIn('inspections', session.read_output())
                session.end_turn()
                self.assertEqual(len(session.inspections), 1)
                self.assertEqual(session.inspections[0]['turn'], 3)
                session.inspect_namespace()
                self.assertEqual(len(session.inspections), 1)
            finally:
                experiment.stream.close()

    def test_trace_observes_identical_http_payload_and_redacts_key(self):
        context = model_context(ModelContext(QUESTION, 1, 1, 1, 20, (), (), ()), [], '/project', 98304)
        payloads, trace = [], []
        for callback in (None, trace.append):
            model = DeepSeekFlash('secret-key', trace=callback)
            with patch('nervus.models.deepseek.request.urlopen', return_value=response('{"type":"finish","answer":"secret-key"}')) as post:
                self.assertEqual(model.decide(context), Finish('secret-key'))
            payloads.append(post.call_args.args[0].data)
        self.assertEqual(payloads[0], payloads[1])
        self.assertEqual(trace[0]['body'], json.loads(payloads[1]))
        self.assertNotIn('tool_choice', trace[0]['body'])
        self.assertNotIn('secret-key', json.dumps(trace))
        self.assertNotIn('Authorization', json.dumps(trace))
        self.assertIn('<REDACTED>', trace[1]['body']['choices'][0]['message']['content'])

    def test_recording_failure_does_not_change_finish(self):
        def broken(event):
            raise OSError('disk failure')
        model = DeepSeekFlash('secret-key', trace=broken)
        with patch('nervus.models.deepseek.request.urlopen', return_value=response('{"type":"finish","answer":"original"}')):
            self.assertEqual(model.decide(ModelContext(QUESTION, 1, 1, 1, 20, (), (), ())), Finish('original'))
        self.assertEqual(model.trace_errors, ['OSError', 'OSError'])

    def test_free_code_choices_and_real_inspection_are_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / 'code-trace.jsonl'
            turns = [
                [{'type': 'code', 'code': 'if False: inspect_workspace()', 'exports': []},
                 {'type': 'finish', 'answer': 'run echo add upper; 4'}],
                [{'type': 'code', 'code': 'catalog = inspect_workspace()', 'exports': ['catalog']},
                 {'type': 'finish', 'answer': 'run echo add; 3'}],
                [{'type': 'finish', 'answer': 'run echo add upper; 4'}],
            ]
            terminal = Terminal(root, turns, '--cap-log', str(log))
            try:
                for count in (4, 3, 4):
                    terminal.send(f'/cap-env {count}')
                    terminal.expect('你> ')
                    terminal.send('/cap-ask')
                    terminal.expect('实验记录已保存')
                    terminal.expect('你> ')
            finally:
                self.assertEqual(terminal.close(), 0)
            events = [json.loads(line) for line in log.read_text().splitlines()]
            finished = [e for e in events if e['event'] == 'trial_finished']
            self.assertEqual([e['inspection']['code_inspection'] for e in finished], ['needs_review', 'yes', 'no_code'])
            self.assertEqual(finished[1]['inspection']['workspace_api_calls'][0]['turn'], 2)
            self.assertEqual([e['action_type'] for e in events if e['event'] == 'model_action'], ['Code', 'Finish', 'Code', 'Finish', 'Finish'])
