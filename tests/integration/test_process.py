"""Real local processes with barriers at the supervisor's creation boundary."""
from functools import partial
import os
import signal
import sys
import tempfile
from pathlib import Path
import threading
import time
import unittest
from unittest.mock import patch

from nervus import Capability, Code, Finish, ProcessRunner, ScriptedModel, Session, SessionStateError, WorkingStateLostError


def wait_for(predicate):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.005)
    raise AssertionError('Barrier timed out')


class WorkerIdentity:
    async def initialize(self):
        pass
    async def __call__(self):
        return os.getpid()
    async def close(self):
        pass


class ProcessTests(unittest.TestCase):
    def launch(self, session, code):
        results, errors = [], []
        def run():
            try:
                results.append(session.run('run process', ScriptedModel([Code(code, exports=('result',)), Finish('done')])))
            except Exception as error:
                errors.append(error)
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return thread, results, errors

    def code(self, source):
        return 'saved = 42\nresult = await tools.run(' + repr([sys.executable, '-c', source]) + ')'

    def assert_gone(self, session):
        running = [e for e in session.process_events if e['event'] == 'running']
        self.assertTrue(running)
        for event in running:
            with self.assertRaises(ProcessLookupError):
                os.killpg(event['pid'], 0)

    def test_worker_death_seals_call_before_supervisor_finishes_recovery(self):
        gate = threading.Event()
        with Session() as session:
            session.publish([Capability('run', ProcessRunner)])
            owner = session._runtime.processes
            collect = owner._collect
            def paused(start):
                if not gate.wait(3):
                    raise RuntimeError('cleanup barrier timeout')
                return collect(start)
            with patch.object(owner, '_collect', paused):
                thread, results, errors = self.launch(session, self.code('import time; time.sleep(60)'))
                try:
                    wait_for(lambda: any(e['event'] == 'running' for e in session.process_events))
                    os.kill(session._runtime._process.pid, signal.SIGKILL)
                    wait_for(lambda: any(e['event'] == 'worker_exit_confirmed' for e in session.process_events))
                    self.assertEqual(session.calls[0]['status'], 'interrupted')
                    self.assertEqual(session.calls[0]['outcome'], 'unknown')
                    self.assertTrue(thread.is_alive())
                    with self.assertRaises(SessionStateError):
                        session.begin_turn()
                finally:
                    gate.set()
                    thread.join(4)
            self.assertIsInstance(errors[0], WorkingStateLostError)
            self.assert_gone(session)
            events = [e['event'] for e in session.process_events]
            self.assertLess(events.index('worker_exit_confirmed'), events.index('exited'))
            self.assertEqual(session.calls[0]['status'], 'interrupted')
            with self.assertRaises(WorkingStateLostError):
                session.begin_turn()

    def test_stop_owns_request_while_popen_has_not_returned(self):
        entered, release = threading.Event(), threading.Event()
        with Session() as session:
            session.publish([Capability('run', ProcessRunner)])
            owner = session._runtime.processes
            spawn = owner._spawn
            def paused(argv):
                entered.set()
                if not release.wait(3):
                    raise RuntimeError('spawn barrier timeout')
                return spawn(argv)
            with patch.object(owner, '_spawn', paused):
                thread, results, errors = self.launch(session, self.code('import time; time.sleep(60)'))
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
                    wait_for(lambda: owner._closed_turns)
                    with self.assertRaises(SessionStateError):
                        session.begin_turn()
                finally:
                    release.set()
                    stopper.join(4)
                    thread.join(4)
            self.assertEqual(stop_errors, [])
            self.assertEqual(errors, [])
            self.assertTrue(stopped[0]['stopped'])
            self.assertEqual(results[0].reason, 'stopped')
            self.assert_gone(session)
            session.begin_turn()
            self.assertEqual(session.execute('value = saved', exports=['value']), {'value': 42})
            session.end_turn()

    def test_closed_admission_rejects_late_start_without_spawning(self):
        entered, release = threading.Event(), threading.Event()
        with Session() as session:
            session.publish([Capability('run', ProcessRunner)])
            owner = session._runtime.processes
            submit = owner.submit
            def delayed(*args):
                entered.set()
                if not release.wait(3):
                    raise RuntimeError('request barrier timeout')
                return submit(*args)
            with patch.object(owner, 'submit', delayed), patch.object(owner, '_spawn', side_effect=AssertionError('must not spawn')) as spawn:
                thread, results, errors = self.launch(session, self.code('pass'))
                self.assertTrue(entered.wait(3))
                stop_errors = []
                def stop():
                    try:
                        session.stop(session.active_turn, grace=2)
                    except Exception as error:
                        stop_errors.append(error)
                stopper = threading.Thread(target=stop, daemon=True)
                stopper.start()
                try:
                    wait_for(lambda: owner._closed_turns)
                finally:
                    release.set()
                    stopper.join(4)
                    thread.join(4)
                spawn.assert_not_called()
            self.assertEqual(errors, [])
            self.assertEqual(stop_errors, [])
            self.assertFalse(any(e['event'] == 'accepted' for e in session.process_events))

    def test_exit_code_argv_and_bounded_dual_streams_use_normal_call_pipeline(self):
        with Session() as session:
            capability = Capability('run', partial(ProcessRunner, output_limit=123), implementation_version='process-v1')
            session.publish([capability])
            source = "import os,sys; os.write(1, b'a'*200000); os.write(2, b'b'*200000); sys.exit(7)"
            result = session.run('outputs', ScriptedModel([Code(self.code(source), exports=('result',)), Finish('done')]))
            value = result.feedback[0].values['result']
            self.assertEqual(value, {'returncode': 7, 'stdout': 'a'*123, 'stderr': 'b'*123,
                                     'stdout_truncated': True, 'stderr_truncated': True})
            self.assertEqual([(c['turn'], c['implementation_version'], c['status']) for c in session.calls], [(1, 'process-v1', 'succeeded')])
            self.assertTrue(all(e['identity'] == capability.identity for e in session.process_events))
            self.assert_gone(session)

    def test_normal_exit_racing_stop_keeps_exit_fact_separate_from_call(self):
        exited, release = threading.Event(), threading.Event()
        with Session() as session:
            session.publish([Capability('run', ProcessRunner)])
            owner = session._runtime.processes
            record = owner._record
            def delayed(start, event, **details):
                if event == 'exited':
                    exited.set()
                    if not release.wait(3):
                        raise RuntimeError('exit record barrier timeout')
                return record(start, event, **details)
            with patch.object(owner, '_record', delayed):
                thread, results, errors = self.launch(session, self.code('pass'))
                self.assertTrue(exited.wait(3))
                stop_errors = []
                def stop():
                    try:
                        session.stop(session.active_turn, grace=2)
                    except Exception as error:
                        stop_errors.append(error)
                stopper = threading.Thread(target=stop, daemon=True)
                stopper.start()
                try:
                    wait_for(lambda: owner._closed_turns)
                finally:
                    release.set()
                    stopper.join(4)
                    thread.join(4)
            self.assertEqual(errors, [])
            self.assertEqual(stop_errors, [])
            self.assertEqual(results[0].reason, 'stopped')
            self.assertEqual(session.calls[0]['status'], 'cancelled')
            self.assertEqual([e['returncode'] for e in session.process_events if e['event'] == 'exited'], [0])
            self.assert_gone(session)

    def test_process_capability_obeys_alias_snapshot_budget_and_argv_rules(self):
        from dataclasses import replace
        from nervus import CodeExecutionError
        with Session() as session:
            old = Capability('run', partial(ProcessRunner, output_limit=8), implementation_version='old')
            session.publish([old])
            session.begin_turn(call_budget=2)
            session.execute('runner = tools.run')
            session.publish([replace(old, factory=partial(ProcessRunner, output_limit=3), implementation_version='new')])
            argv = [sys.executable, '-c', 'import sys; print(sys.argv[1],end="")', 'a; $HOME']
            self.assertEqual(session.execute(f'result = await runner({argv!r})', exports=['result'])['result']['stdout'], 'a; $HOME')
            with self.assertRaisesRegex(CodeExecutionError, 'argv must'):
                session.execute("await runner('not an argv list')")
            with self.assertRaisesRegex(CodeExecutionError, 'budget exhausted'):
                session.execute(f'await runner({argv!r})')
            session.end_turn()
            session.begin_turn()
            with self.assertRaises(SessionStateError):
                session._runtime.request('end', target_turn=1)
            self.assertEqual(session.execute(f'result = await runner({argv!r})', exports=['result'])['result']['stdout'], 'a; ')
            session.end_turn()
            self.assertEqual([c['implementation_version'] for c in session.calls], ['old', 'old', 'new'])
            self.assertEqual(len([e for e in session.process_events if e['event'] == 'accepted']), 2)
            self.assert_gone(session)

    def test_start_failure_is_reported_without_poisoning_next_turn(self):
        from nervus import CodeExecutionError
        with Session() as session:
            session.publish([Capability('run', ProcessRunner)])
            session.begin_turn()
            with self.assertRaisesRegex(CodeExecutionError, 'FileNotFoundError'):
                session.execute("await tools.run(['/nervus-nonexistent-command'])")
            session.end_turn()
            self.assertEqual(session.calls[0]['status'], 'failed')
            self.assertEqual(session.process_events[-1]['event'], 'start_failed')
            session.begin_turn()
            session.execute(self.code('pass'))
            session.end_turn()

    def test_parallel_processes_drain_both_streams(self):
        with Session() as session:
            session.publish([Capability('run', partial(ProcessRunner, output_limit=10))])
            source = "import os,threading; t=threading.Thread(target=os.write,args=(1,b'a'*200000)); t.start(); os.write(2,b'b'*200000); t.join()"
            argv = [sys.executable, '-c', source]
            session.begin_turn()
            value = session.execute(f'import asyncio\nresults = await asyncio.gather(tools.run({argv!r}), tools.run({argv!r}))', exports=['results'])
            for result in value['results']:
                self.assertEqual(result['returncode'], 0)
                self.assertEqual(result['stdout'], 'a'*10)
                self.assertEqual(result['stderr'], 'b'*10)
                self.assertTrue(result['stdout_truncated'] and result['stderr_truncated'])
            session.end_turn()
            self.assertEqual(len(session.calls), 2)
            self.assert_gone(session)

    def test_only_process_resources_move_to_supervisor(self):
        with Session() as session:
            session.publish([Capability('run', ProcessRunner), Capability('identity', WorkerIdentity)])
            session.begin_turn()
            values = session.execute(self.code('import os; print(os.getppid())') + '\nworker = await tools.identity()', exports=['result', 'worker'])
            self.assertEqual(int(values['result']['stdout']), os.getpid())
            self.assertEqual(values['worker'], session._runtime._process.pid)
            session.end_turn()

    def test_stop_signals_group_and_waits_for_descendant_exit(self):
        with tempfile.TemporaryDirectory() as directory, Session() as session:
            ready = Path(directory) / 'ready'
            child_source = 'import time; time.sleep(60)'
            source = (
                'import subprocess,signal,sys,time,pathlib,os\n'
                f'child = subprocess.Popen([sys.executable, "-c", {child_source!r}])\n'
                'def stop(*args):\n child.wait()\n sys.exit(0)\n'
                'signal.signal(signal.SIGTERM, stop)\n'
                f'pathlib.Path({str(ready)!r}).write_text(str(child.pid))\n'
                'while True: time.sleep(1)\n'
            )
            session.publish([Capability('run', ProcessRunner)])
            thread, results, errors = self.launch(session, self.code(source))
            wait_for(lambda: ready.exists() and ready.read_text())
            descendant = int(ready.read_text())
            session.stop(session.active_turn, grace=2)
            thread.join(3)
            self.assertEqual(errors, [])
            self.assertEqual(results[0].reason, 'stopped')
            self.assert_gone(session)
            with self.assertRaises(ProcessLookupError):
                os.kill(descendant, 0)

    def test_ignoring_term_escalates_to_group_kill(self):
        with tempfile.TemporaryDirectory() as directory, Session() as session:
            ready = Path(directory) / 'ready'
            source = ("import signal,time,pathlib; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                      f"pathlib.Path({str(ready)!r}).touch(); time.sleep(60)")
            session.publish([Capability('run', ProcessRunner)])
            thread, results, errors = self.launch(session, self.code(source))
            wait_for(ready.exists)
            session.stop(session.active_turn, grace=2)
            thread.join(3)
            self.assertEqual(errors, [])
            self.assert_gone(session)
            self.assertEqual([e['returncode'] for e in session.process_events if e['event'] == 'exited'], [-signal.SIGKILL])

    def test_unconfirmed_cleanup_blocks_next_turn_and_is_not_reported_as_success(self):
        from nervus import ProcessCleanupError
        session = Session()
        session.publish([Capability('run', ProcessRunner)])
        owner = session._runtime.processes
        collect = owner._collect
        def cannot_confirm(start):
            collect(start)  # Real process is stopped; inject a confirmation failure.
            raise ProcessCleanupError('injected confirmation failure')
        try:
            with patch.object(owner, '_collect', cannot_confirm):
                thread, results, errors = self.launch(session, self.code('pass'))
                thread.join(3)
            self.assertEqual(results, [])
            self.assertIsInstance(errors[0], ProcessCleanupError)
            self.assertEqual(session.process_events[-1]['event'], 'cleanup_unconfirmed')
            with self.assertRaises(ProcessCleanupError):
                session.begin_turn()
        finally:
            with self.assertRaises(ProcessCleanupError):
                session.close()
        self.assertFalse(session._runtime._process.is_alive())

    def test_failed_stop_still_allows_worker_close_without_claiming_cleanup_success(self):
        from nervus import ProcessCleanupError
        entered, release = threading.Event(), threading.Event()
        session = Session()
        session.publish([Capability('run', ProcessRunner)])
        owner = session._runtime.processes
        collect = owner._collect
        def cannot_confirm(start):
            collect(start)
            entered.set()
            if not release.wait(3):
                raise RuntimeError('confirmation barrier timeout')
            raise ProcessCleanupError('injected confirmation failure')
        with patch.object(owner, '_collect', cannot_confirm):
            thread, results, errors = self.launch(session, self.code('pass'))
            self.assertTrue(entered.wait(3))
            stop_errors = []
            def stop():
                try:
                    session.stop(session.active_turn, grace=2)
                except Exception as error:
                    stop_errors.append(error)
            stopper = threading.Thread(target=stop, daemon=True)
            stopper.start()
            try:
                wait_for(lambda: owner._closed_turns)
            finally:
                release.set()
                stopper.join(4)
                thread.join(4)
        try:
            self.assertIsInstance(stop_errors[0], ProcessCleanupError)
            self.assertEqual(results, [])
            with self.assertRaises(SessionStateError):
                session.begin_turn()
            with self.assertRaises(ProcessCleanupError):
                session.close()
            with self.assertRaises(ProcessCleanupError):
                session.close()
            self.assertFalse(session._runtime._process.is_alive())
        finally:
            # Test-only last resort if the public close assertion fails.
            if session._runtime._process.is_alive():
                session._runtime._join(terminate=True)
