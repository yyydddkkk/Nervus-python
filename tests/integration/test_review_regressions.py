"""Regression cases from the whole-kernel review; no model API calls."""

from dataclasses import replace
import threading
import unittest

from nervus import Capability, Code, Finish, Session, SessionStateError, WorkingStateLostError
from tests.support.capabilities import Echo
from tests.support.review_cases import ( close_after_retirement_failure, falsey_task_exception,
    late_terminal_after_confirmed_exit,
)


class ReviewRegressions(unittest.TestCase):
    def test_model_turn_rejects_manual_control_but_allows_publish(self):
        entered, release = threading.Event(), threading.Event()
        results, errors = [], []

        class Model:
            def decide(self, context):
                if context.step == 1:
                    entered.set()
                    if not release.wait(5):
                        raise RuntimeError('barrier timed out')
                    return Code("value = await tools.read('input')", exports=('value',))
                return Finish('done')

        with Session() as session:
            cap = Capability('read', Echo, implementation_version='old')
            session.publish([cap])
            def run():
                try:
                    results.append(session.run('input', Model()))
                except Exception as error:
                    errors.append(error)
            runner = threading.Thread(target=run)
            runner.start()
            try:
                self.assertTrue(entered.wait(3))
                for operation in (session.end_turn, session.begin_turn, session.describe_turn,
                                  session.read_output, session.inspect_namespace,
                                  lambda: session.execute('wrong = True'), session.close):
                    with self.assertRaises(SessionStateError):
                        operation()
                session.publish([replace(cap, factory=Echo, implementation_version='new')])
            finally:
                release.set()
                runner.join(5)
            self.assertFalse(runner.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(results[0].feedback[0].values['value'], 'input')
            session.begin_turn()
            self.assertEqual(session.execute("value = await tools.read('next')", exports=['value'])['value'], 'next')
            session.end_turn()
            self.assertEqual([(c['turn'], c['implementation_version']) for c in session.calls],
                             [(1, 'old'), (2, 'new')])

    def test_close_continues_after_retirement_failure(self):
        result = close_after_retirement_failure()
        self.assertIsNotNone(result['error'])
        self.assertCountEqual(result['close_hooks_called'], ['old', 'new'])
        self.assertTrue(all(e['verified'] and e['close_count'] == 1 for e in result['closed']))

    def test_exit_confirmation_precedes_late_terminal(self):
        result = late_terminal_after_confirmed_exit()
        self.assertEqual(result['calls'][0]['status'], 'interrupted')
        self.assertEqual(result['calls'][0]['outcome'], 'unknown')
        self.assertIn('terminal_rejected', result['order'])

    def test_falsey_exception_is_reported_as_failure(self):
        result = falsey_task_exception()
        self.assertEqual(result['task_report'][0]['status'], 'failed')
        self.assertIn('QuietFailure: background failure', result['task_report'][0]['error'])

    def test_accepted_terminal_survives_exit_and_duplicate_facts(self):
        with Session(execution_timeout=0.5) as session:
            session.publish([Capability('echo', Echo)])
            session.begin_turn()
            session.execute("result = await tools.echo('confirmed')")
            original, = session.calls
            # Replay through the fact owner's boundary, never through the model.
            self.assertFalse(session._runtime.journal.accept({**original, 'result': 'duplicate'}))
            with self.assertRaises(WorkingStateLostError):
                session.execute('while True: pass')
            self.assertFalse(session._runtime.journal.accept({**original, 'result': 'late'}))
            self.assertEqual(session.calls, (original,))

    def test_target_turn_validation_rejects_stale_execution_reads_and_end(self):
        with Session() as session:
            first = session.begin_turn()['turn']
            session.execute('saved = 42')
            session.end_turn()
            second = session.begin_turn()['turn']
            # Exercise the worker protocol check separately from the Host guard.
            for command, arguments in (
                ('execute', {'code': 'saved = 0', 'exports': ()}),
                ('describe', {}), ('output', {}), ('end', {}),
            ):
                with self.assertRaisesRegex(SessionStateError, 'Target Turn'):
                    session._runtime.request(command, target_turn=first, **arguments)
            self.assertEqual(session.describe_turn()['turn'], second)
            self.assertEqual(session.execute('copy = saved', exports=['copy']), {'copy': 42})
            session.end_turn()

    def test_reentrant_model_cannot_manually_end_its_turn(self):
        with Session() as session:
            class Model:
                def decide(inner, context):
                    with self.assertRaises(SessionStateError):
                        session.end_turn()
                    with self.assertRaises(SessionStateError):
                        session.run('nested', inner)
                    return Finish('done')
            self.assertEqual(session.run('input', Model()).answer, 'done')
            session.begin_turn()
            session.end_turn()

    def test_close_reports_both_retired_and_published_failures(self):
        result = close_after_retirement_failure(fail_new=True)
        self.assertCountEqual(result['close_hooks_called'], ['old', 'new'])
        self.assertTrue(all(e['verified'] and e['close_count'] == 1 for e in result['closed']))
        self.assertIn('read: RuntimeError', result['error'])
        self.assertIn('replacement: RuntimeError', result['error'])
