"""Host stop is bound to a Turn and independent of the executing command."""
from functools import partial
from dataclasses import replace
from pathlib import Path
import tempfile
import asyncio
import multiprocessing
import threading
import unittest

from tests.support.capabilities import SQLiteQuery

from nervus import Capability, Code, Finish, ScriptedModel, Session, SessionStateError, WorkingStateLostError


class BlockingCapability:
    def __init__(self, entered, cleaned, release, mode):
        self.entered, self.cleaned, self.release, self.mode = entered, cleaned, release, mode

    async def initialize(self):
        pass

    async def __call__(self):
        self.entered.set()
        if self.mode == 'spin':
            while True:
                pass
        if self.mode == 'ignore':
            while True:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.cleaned.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.cleaned.set()
            if self.mode == 'cleanup_spin':
                while True:
                    pass
            if self.mode == 'cleanup_barrier':
                while not self.release.is_set():
                    await asyncio.sleep(0.001)

    async def close(self):
        pass


class StopTests(unittest.TestCase):
    def launch(self, session, model):
        results, errors = [], []
        def run():
            try:
                results.append(session.run('work', model))
            except Exception as error:
                errors.append(error)
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return thread, results, errors

    def test_stop_running_entry_preserves_state_and_finishes_tasks(self):
        with multiprocessing.Manager() as manager, Session(execution_timeout=5) as session:
            entered, cleaned, release = manager.Event(), manager.Event(), manager.Event()
            session.publish([Capability('wait', partial(BlockingCapability, entered, cleaned, release, 'cooperate'))])
            model = ScriptedModel([Code("saved = 41\ndef helper(): return saved + 1\nawait tools.wait()"), Finish('late')])
            thread, results, errors = self.launch(session, model)
            self.assertTrue(entered.wait(3))
            turn = session.active_turn
            report = session.stop(turn, grace=1)
            thread.join(3)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            self.assertTrue(report['stopped'])
            self.assertFalse(report['working_state_lost'])
            self.assertTrue(cleaned.is_set())
            self.assertEqual(results[0].reason, 'stopped')
            self.assertEqual(len(model.requests), 1)
            self.assertEqual(session.calls[0]['status'], 'cancelled')
            session.begin_turn()
            self.assertEqual(session.execute('value = helper()', exports=['value']), {'value': 42})
            session.end_turn()
            session.close()
            self.assertEqual(session._runtime._process.exitcode, 0)

    def test_late_decision_cannot_execute_or_close_new_turn(self):
        for action in (Code('saved = 0'), Finish('late'), RuntimeError('late error')):
            with self.subTest(action=action), Session() as session:
                entered, release = threading.Event(), threading.Event()
                class Paused:
                    def decide(inner, context):
                        entered.set()
                        if not release.wait(5):
                            raise RuntimeError('barrier timeout')
                        if isinstance(action, Exception):
                            raise action
                        return action
                thread, results, errors = self.launch(session, Paused())
                try:
                    self.assertTrue(entered.wait(3))
                    turn = session.active_turn
                    session.stop(turn)
                    self.assertTrue(thread.is_alive())  # decide itself was not interrupted.
                    second = session.begin_turn()['turn']
                    session.execute('saved = 42')
                    with self.assertRaises(SessionStateError):
                        session.stop(turn)
                finally:
                    release.set()
                    thread.join(3)
                self.assertEqual(errors, [])
                self.assertEqual(results[0].reason, 'stopped')
                self.assertEqual(results[0].decisions, 1)
                self.assertEqual(session.describe_turn()['turn'], second)
                self.assertEqual(session.execute('value = saved', exports=['value']), {'value': 42})
                session.end_turn()

    def test_new_turn_cannot_pass_pending_cleanup(self):
        with multiprocessing.Manager() as manager, Session(execution_timeout=5) as session:
            entered, cleaned, release = manager.Event(), manager.Event(), manager.Event()
            session.publish([Capability('wait', partial(BlockingCapability, entered, cleaned, release, 'cleanup_barrier'))])
            thread, results, errors = self.launch(session, ScriptedModel([Code('await tools.wait()')]))
            self.assertTrue(entered.wait(3))
            stopped, stop_errors = [], []
            def stop():
                try:
                    stopped.append(session.stop(session.active_turn, grace=2))
                except Exception as error:
                    stop_errors.append(error)
            stopper = threading.Thread(target=stop, daemon=True)
            stopper.start()
            try:
                self.assertTrue(cleaned.wait(3))
                with self.assertRaises(SessionStateError):
                    session.begin_turn()
            finally:
                release.set()
                stopper.join(3)
                thread.join(3)
            self.assertEqual(stop_errors, [])
            self.assertEqual(errors, [])
            self.assertTrue(stopped[0]['stopped'])
            session.begin_turn()
            session.end_turn()

    def test_uncooperative_stop_confirms_exit_and_reports_state_loss(self):
        for mode in ('spin', 'ignore', 'cleanup_spin'):
            with self.subTest(mode=mode), multiprocessing.Manager() as manager, Session(execution_timeout=5) as session:
                entered, cleaned, release = manager.Event(), manager.Event(), manager.Event()
                session.publish([Capability('wait', partial(BlockingCapability, entered, cleaned, release, mode))])
                thread, results, errors = self.launch(session, ScriptedModel([Code('saved = 42\nawait tools.wait()')]))
                self.assertTrue(entered.wait(3))
                with self.assertRaisesRegex(WorkingStateLostError, 'Worker exit confirmed'):
                    session.stop(session.active_turn, grace=0.2)
                thread.join(3)
                self.assertFalse(thread.is_alive())
                self.assertIsInstance(errors[0], WorkingStateLostError)
                self.assertFalse(session._runtime._process.is_alive())
                self.assertIsNotNone(session._runtime._process.exitcode)
                self.assertEqual(session.calls[0]['status'], 'interrupted')
                self.assertEqual(session.calls[0]['outcome'], 'unknown')
                with self.assertRaises(WorkingStateLostError):
                    session.begin_turn()

    def test_stop_drains_background_task_while_model_is_pending(self):
        entered, release = threading.Event(), threading.Event()
        with Session() as session:
            def pause(context):
                entered.set()
                if not release.wait(5):
                    raise RuntimeError('barrier timeout')
                return Finish('late')
            model = ScriptedModel([Code('''import asyncio
saved = []
started = asyncio.Event()
async def work():
    started.set()
    try:
        await asyncio.Event().wait()
    finally:
        saved.append('cleaned')
task = asyncio.create_task(work())
await started.wait()
'''), pause])
            thread, results, errors = self.launch(session, model)
            try:
                self.assertTrue(entered.wait(3))
                turn = session.active_turn
                with self.assertRaises(SessionStateError):
                    session.stop(turn + 1)
                report = session.stop(turn)
                self.assertEqual(report['tasks'][0]['status'], 'cancelled')
                self.assertEqual(report['tasks'][0]['turn'], turn)
                session.begin_turn()
                session.execute("assert task.done()\nassert saved == ['cleaned']")
                session.end_turn()
            finally:
                release.set()
                thread.join(3)
            self.assertEqual(errors, [])
            self.assertEqual(results[0].reason, 'stopped')

    def test_late_old_run_cannot_release_new_model_control(self):
        old_entered, old_release = threading.Event(), threading.Event()
        new_entered, new_release = threading.Event(), threading.Event()
        def paused(entered, release):
            def decide(context):
                entered.set()
                if not release.wait(5):
                    raise RuntimeError('barrier timeout')
                return Finish('done')
            return ScriptedModel([decide])
        with Session() as session:
            old_thread, old_results, old_errors = self.launch(session, paused(old_entered, old_release))
            new_thread = None
            try:
                self.assertTrue(old_entered.wait(3))
                session.stop(session.active_turn)
                new_thread, new_results, new_errors = self.launch(session, paused(new_entered, new_release))
                self.assertTrue(new_entered.wait(3))
                current = session.active_turn
                old_release.set()
                old_thread.join(3)
                self.assertEqual(old_errors, [])
                self.assertEqual(old_results[0].reason, 'stopped')
                self.assertEqual(session.active_turn, current)
                with self.assertRaises(SessionStateError):
                    session.end_turn()
            finally:
                old_release.set()
                new_release.set()
                old_thread.join(3)
                if new_thread:
                    new_thread.join(3)
            self.assertEqual(new_errors, [])
            self.assertEqual(new_results[0].reason, 'finished')

    def test_stop_reports_retirement_failure_after_execution_has_stopped(self):
        entered, release = threading.Event(), threading.Event()
        with tempfile.TemporaryDirectory() as directory, Session() as session:
            audit = str(Path(directory) / 'close.jsonl')
            cap = Capability('read', partial(SQLiteQuery, audit, 'old', fail_close=True))
            session.publish([cap])
            def pause(context):
                entered.set()
                if not release.wait(5):
                    raise RuntimeError('barrier timeout')
                return Finish('late')
            thread, results, errors = self.launch(session, ScriptedModel([Code('saved = 42'), pause]))
            try:
                self.assertTrue(entered.wait(3))
                session.publish([replace(cap, factory=partial(SQLiteQuery, audit, 'new'))])
                report = session.stop(session.active_turn)
                self.assertTrue(report['stopped'])
                self.assertIn('injected release failure', report['release_errors'][0])
                session.begin_turn()
                self.assertEqual(session.execute('value = await tools.read(saved)', exports=['value'])['value']['marker'], 'new')
                session.end_turn()
            finally:
                release.set()
                thread.join(3)
            self.assertEqual(errors, [])
            self.assertEqual(results[0].reason, 'stopped')
