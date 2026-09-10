"""Offline Host checks; no real model API."""
import os
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import queue
import signal
import tempfile
import threading
import unittest
from unittest.mock import patch

from examples.chat_session import Job, load_env, main
from nervus import Code, Finish, ScriptedModel, Session


class ChatSessionTests(unittest.TestCase):
    def test_help_needs_no_key(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(SystemExit) as exit:
            main(['--help'])
        self.assertEqual(exit.exception.code, 0)

    def test_env_is_literal_and_environment_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / '.env'
            path.write_text('CHAT_TEST=from-file\nexport CHAT_OTHER="$(do-not-run)"\n')
            with patch.dict(os.environ, {'CHAT_TEST': 'existing'}, clear=True):
                load_env(path)
                self.assertEqual(os.environ['CHAT_TEST'], 'existing')
                self.assertEqual(os.environ['CHAT_OTHER'], '$(do-not-run)')

    def test_quit_and_eof_close_without_waiting_for_model(self):
        original_stop, original_close = Session.stop, Session.close
        def noisy_stop(session, *args, **kwargs):
            signal.raise_signal(signal.SIGINT)
            signal.raise_signal(signal.SIGINT)
            return original_stop(session, *args, **kwargs)
        def noisy_close(session):
            signal.raise_signal(signal.SIGINT)
            return original_close(session)
        original_handler = signal.getsignal(signal.SIGINT)
        for command in ('/quit', None):
            with self.subTest(command=command):
                entered, release = threading.Event(), threading.Event()
                def blocked(context):
                    entered.set()
                    release.wait(5)
                    return Finish('late answer')
                def inputs(events):
                    events.put((None, 'input', 'task'))
                    if entered.wait(5):
                        events.put((None, 'quit' if command is None else 'input', command))
                model = ScriptedModel([blocked])
                try:
                    with patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'offline-placeholder'}), \
                         patch('examples.chat_session.DeepSeekFlash', return_value=model), \
                         patch('examples.chat_session.read_input', side_effect=inputs), \
                         patch.object(Session, 'stop', noisy_stop), \
                         patch.object(Session, 'close', noisy_close), \
                         patch('urllib.request.urlopen', side_effect=AssertionError('Network forbidden')):
                        self.assertEqual(main([]), 0)
                        self.assertIs(signal.getsignal(signal.SIGINT), original_handler)
                        self.assertTrue(entered.is_set())
                        self.assertFalse(release.is_set())
                finally:
                    release.set()

    def test_constructor_failure_restores_signal_handler(self):
        handler = signal.getsignal(signal.SIGINT)
        output = StringIO()
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'offline-placeholder'}),              patch('examples.chat_session.Session', side_effect=RuntimeError('private details')),              redirect_stdout(output):
            self.assertEqual(main([]), 1)
        self.assertIs(signal.getsignal(signal.SIGINT), handler)
        self.assertNotIn('Session closed.', output.getvalue())
        self.assertNotIn('private details', output.getvalue())

    def test_cleanup_failure_returns_nonzero_without_claiming_closed(self):
        handler = signal.getsignal(signal.SIGINT)
        output = StringIO()
        def quit_input(events):
            events.put((None, 'quit', None))
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'offline-placeholder'}),              patch('examples.chat_session.Session') as factory,              patch('examples.chat_session.read_input', side_effect=quit_input),              redirect_stdout(output):
            factory.return_value.close.side_effect = RuntimeError('private details')
            self.assertEqual(main([]), 1)
        self.assertIs(signal.getsignal(signal.SIGINT), handler)
        self.assertIn('cleanup-error', output.getvalue())
        self.assertNotIn('Session closed.', output.getvalue())
        self.assertNotIn('private details', output.getvalue())

    def test_state_and_stale_response(self):
        events = queue.Queue()
        entered, release = threading.Event(), threading.Event()
        def blocked(context):
            entered.set()
            release.wait(5)
            return Code('stale = True')
        with Session() as session:
            first = Job(session, ScriptedModel([Code('saved = 42'), blocked]),
                        'save', events, 4, 4)
            first.thread.start()
            self.assertTrue(entered.wait(5))
            self.assertTrue(first.thread.daemon)
            self.assertTrue(first.stop())
            self.assertTrue(first.thread.is_alive())
            result = session.run('reuse', ScriptedModel([
                Code('assert saved == 42; assert "stale" not in globals()'), Finish('ok')]))
            self.assertEqual(result.answer, 'ok')
            release.set()
            first.thread.join(5)
            self.assertFalse(first.thread.is_alive())
            emitted = []
            while not events.empty():
                emitted.append(events.get_nowait())
            self.assertFalse(any(kind == 'code' and value == 'stale = True'
                                 for _, kind, value in emitted))
            self.assertTrue(any(kind == 'result' and value.answer is None
                                and value.reason == 'stopped' for _, kind, value in emitted))


if __name__ == '__main__':
    unittest.main()
