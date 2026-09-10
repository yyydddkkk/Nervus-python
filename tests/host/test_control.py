from contextlib import redirect_stdout
from io import StringIO
import os
from pathlib import Path
import signal
import tempfile
import threading
import unittest
from unittest.mock import patch

from nervus import Finish, ScriptedModel, Session
from nervus.host import terminal
from nervus.host.history import History


class ControlTests(unittest.TestCase):
    def test_help_needs_no_credentials(self):
        with patch.dict(os.environ, {}, clear=True), redirect_stdout(StringIO()), self.assertRaises(SystemExit) as error:
            terminal.main(['--help'])
        self.assertEqual(error.exception.code, 0)

    def test_quit_does_not_wait_for_model_and_restores_signal_handler(self):
        entered, release = threading.Event(), threading.Event()
        def blocked(context):
            entered.set()
            release.wait(5)
            return Finish('stale')
        def inputs(events):
            events.put((None, 'input', 'task'))
            if entered.wait(3):
                events.put((None, 'input', '/quit'))
        handler = signal.getsignal(signal.SIGINT)
        original_stop, original_close = Session.stop, Session.close
        def stop(session, *args, **kwargs):
            signal.raise_signal(signal.SIGINT)
            return original_stop(session, *args, **kwargs)
        def close(session):
            signal.raise_signal(signal.SIGINT)
            return original_close(session)
        try:
            with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'offline-test'}), \
                 patch.object(terminal, 'DeepSeekFlash', return_value=ScriptedModel([blocked])), \
                 patch.object(terminal, 'read_input', side_effect=inputs), \
                 patch.object(Session, 'stop', stop), patch.object(Session, 'close', close), \
                 patch('urllib.request.urlopen', side_effect=AssertionError('Network forbidden')), redirect_stdout(StringIO()) as output:
                self.assertEqual(terminal.main(['--cwd', directory]), 0)
                self.assertFalse(release.is_set())
                self.assertNotIn('stale', output.getvalue())
                self.assertIn('Session 已关闭', output.getvalue())
            self.assertIs(signal.getsignal(signal.SIGINT), handler)
        finally:
            release.set()

    def test_new_session_ignores_old_answers_and_history_writes(self):
        entered, release, created, finished = (threading.Event() for _ in range(4))
        def blocked(context):
            entered.set()
            release.wait(5)
            return Finish('old-late-answer')
        original_create, original_finish = terminal.create_session, History.finish
        count = 0
        def create(args):
            nonlocal count
            session = original_create(args)
            count += 1
            if count == 2:
                created.set()
            return session
        def finish(history, text):
            result = original_finish(history, text)
            if text == 'fresh-done':
                finished.set()
            return result
        def inputs(events):
            events.put((None, 'input', 'old-task'))
            if entered.wait(3):
                events.put((None, 'input', '/new'))
            if created.wait(3):
                release.set()
                events.put((None, 'input', 'new-task'))
            if finished.wait(3):
                events.put((None, 'input', '/history'))
            events.put((None, 'input', '/quit'))
        try:
            with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'offline-test'}), \
                 patch.object(terminal, 'DeepSeekFlash', side_effect=[ScriptedModel([blocked]), ScriptedModel([Finish('fresh-done')])]), \
                 patch.object(terminal, 'read_input', side_effect=inputs), patch.object(terminal, 'create_session', side_effect=create), \
                 patch.object(History, 'finish', finish), patch('urllib.request.urlopen', side_effect=AssertionError('Network forbidden')), \
                 redirect_stdout(StringIO()) as output:
                self.assertEqual(terminal.main(['--cwd', directory]), 0)
                self.assertIn('user: new-task', output.getvalue())
                self.assertNotIn('user: old-task', output.getvalue())
                self.assertNotIn('old-late-answer', output.getvalue())
        finally:
            release.set()

    def test_rejected_response_is_visible_only_in_debug_mode(self):
        from nervus.models.deepseek import DeepSeekFlash
        from tests.unit.test_deepseek import response
        content = '{"type":"finish","answer":"RAW-REJECTED-TEXT","extra":true}'
        for debug in (False, True):
            with self.subTest(debug=debug):
                done = threading.Event()
                original_finish = History.finish
                def finish(history, text):
                    result = original_finish(history, text)
                    done.set()
                    return result
                def inputs(events):
                    events.put((None, 'input', '你能做些什么'))
                    done.wait(3)
                    events.put((None, 'input', '/quit'))
                with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'offline-key'}), \
                     patch.object(terminal, 'DeepSeekFlash', return_value=DeepSeekFlash('offline-key', debug_responses=debug)), \
                     patch.object(terminal, 'read_input', side_effect=inputs), patch.object(History, 'finish', finish), \
                     patch('nervus.models.deepseek.request.urlopen', return_value=response(content)) as post, \
                     redirect_stdout(StringIO()) as output:
                    self.assertEqual(terminal.main(['--cwd', directory] + (['--debug'] if debug else [])), 0)
                self.assertEqual(post.call_count, 1)
                self.assertIn('unexpected fields: extra', output.getvalue())
                self.assertEqual('RAW-REJECTED-TEXT' in output.getvalue(), debug)
                self.assertEqual('[debug:rejected-response]' in output.getvalue(), debug)
