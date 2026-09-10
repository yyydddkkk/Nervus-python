"""End to end: a real CLI process, stdin commands, real files and managed programs."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest


class Terminal:
    def __init__(self, root, turns, *flags):
        script = root / 'offline.json'
        script.write_text(json.dumps({'turns': turns}))
        env = {**os.environ, 'DEEPSEEK_API_KEY': 'offline-test', 'DEEPSEEK_BASE_URL': 'https://127.0.0.1:1'}
        self.process = subprocess.Popen([sys.executable, '-m', 'nervus.host.terminal', '--cwd', str(root),
                                        '--offline-script', str(script), *flags],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, start_new_session=True, env=env)
        self.text = ''
        self.cursor = 0
        self.condition = threading.Condition()
        def read():
            while char := self.process.stdout.read(1):
                with self.condition:
                    self.text += char
                    self.condition.notify_all()
        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()
        self.expect('你> ')

    def send(self, text):
        self.process.stdin.write(text + '\n')
        self.process.stdin.flush()

    def expect(self, value):
        with self.condition:
            if not self.condition.wait_for(lambda: value in self.text[self.cursor:], 8):
                raise AssertionError(f'Missing {value!r}; output:\n{self.text}')
            self.cursor = self.text.index(value, self.cursor) + len(value)

    def close(self):
        if self.process.poll() is None:
            self.send('/quit')
        try:
            code = self.process.wait(timeout=8)
            self.reader.join(2)
        finally:
            if self.process.poll() is None:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
            self.process.stdin.close()
            self.process.stdout.close()
        return code


def turn(code, answer):
    return [{'type': 'code', 'code': code, 'exports': ['result']}, {'type': 'finish', 'answer': answer}]


class TerminalTests(unittest.TestCase):
    def test_multiturn_files_cwd_history_clear_and_new_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = ("from pathlib import Path\nimport sys\nsaved = 42\ndef total(): return saved\n"
                     "Path('notes.txt').write_text(str(total()))\n"
                     "result = await tools.run([sys.executable, '-c', 'from pathlib import Path; print(Path.cwd())'])\n"
                     f"assert result['stdout'].strip() == {str(root)!r}")
            second = "assert total() == 42\nassert Path('notes.txt').read_text() == '42'\nPath('notes.txt').write_text('updated')\nresult = total()"
            terminal = Terminal(root, [turn(first, '第一次完成'), turn(second, '第二次完成'),
                                       turn("assert 'saved' not in globals()\nresult = 0", '空环境完成')])
            try:
                terminal.send('创建记录')
                terminal.expect('第一次完成')
                terminal.expect('你> ')
                self.assertEqual((root/'notes.txt').read_text(), '42')
                terminal.send('继续修改')
                terminal.expect('第二次完成')
                terminal.expect('你> ')
                self.assertEqual((root/'notes.txt').read_text(), 'updated')
                terminal.send('/history')
                terminal.expect('user: 创建记录')
                terminal.expect('assistant: 第二次完成')
                terminal.expect('你> ')
                terminal.send('/clear-history')
                terminal.expect('Python 变量、函数和文件不变')
                terminal.expect('你> ')
                terminal.send('/workspace')
                terminal.expect('"name": "saved"')
                terminal.expect('你> ')
                terminal.send('/new')
                terminal.expect('聊天历史、变量和函数已清空；文件保留')
                terminal.expect('你> ')
                terminal.send('检查新环境')
                terminal.expect('空环境完成')
                terminal.expect('你> ')
                self.assertNotIn('[debug:', terminal.text)
                self.assertNotIn('代码反馈：', terminal.text)
                self.assertNotIn("Path('notes.txt')", terminal.text)
            finally:
                self.assertEqual(terminal.close(), 0)

    def test_ctrl_c_uses_cooperative_stop_and_preserves_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = "from pathlib import Path; import time; Path('started').touch(); time.sleep(60)"
            code = f"import sys\nsaved = 42\nresult = await tools.run([sys.executable, '-c', {source!r}])"
            terminal = Terminal(root, [turn(code, '不能出现的答案')])
            try:
                terminal.send('运行长任务')
                terminal.expect('执行 Python')
                deadline = time.monotonic() + 5
                while not (root/'started').exists():
                    if time.monotonic() >= deadline:
                        self.fail('program did not start')
                    time.sleep(0.01)
                os.killpg(terminal.process.pid, signal.SIGINT)  # A real terminal-style group signal.
                terminal.expect('Python 状态保留')
                terminal.expect('你> ')
                terminal.send('/workspace')
                terminal.expect('"name": "saved"')
                terminal.expect('你> ')
                self.assertNotIn('不能出现的答案', terminal.text)
                self.assertNotIn('WorkingStateLost', terminal.text)
            finally:
                self.assertEqual(terminal.close(), 0)

    def test_debug_is_complete_and_history_overflow_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            terminal = Terminal(root, [turn('result = 42\nprint("diagnostic")', 'A'*200), turn('result = 43', 'done')],
                                '--debug', '--history-bytes', '150')
            try:
                terminal.send('task')
                terminal.expect('历史已超预算')
                terminal.expect('你> ')
                self.assertIn('"code": "result = 42', terminal.text)
                self.assertIn('[debug:feedback]', terminal.text)
                self.assertIn('diagnostic', terminal.text)
                self.assertIn('A'*200, terminal.text)
                terminal.send('next')
                terminal.expect('消息尚未提交')
                terminal.expect('你> ')
                terminal.send('/clear-history')
                terminal.expect('你> ')
                terminal.send('next')
                terminal.expect('\ndone\n')
                terminal.expect('你> ')
            finally:
                self.assertEqual(terminal.close(), 0)

    def test_worker_loss_requires_explicit_new_session(self):
        with tempfile.TemporaryDirectory() as directory:
            terminal = Terminal(Path(directory), [turn('saved = 42\nwhile True: pass', 'never'),
                turn("assert 'saved' not in globals()\nresult = 1", 'new session works')], '--execution-timeout', '0.5')
            try:
                terminal.send('loop')
                terminal.expect('WorkingStateLostError')
                terminal.expect('你> ')
                terminal.send('continue')
                terminal.expect('Session 不可用，请 /new。任务未提交。')
                terminal.expect('你> ')
                terminal.send('/new')
                terminal.expect('新 Session 已创建')
                terminal.expect('你> ')
                terminal.send('fresh')
                terminal.expect('new session works')
                terminal.expect('你> ')
            finally:
                self.assertEqual(terminal.close(), 0)

    def test_python_visibility_option_shows_code_feedback_and_output_without_protocol(self):
        actions = [
            {'type': 'code', 'code': "print('before-error')\n1 / 0", 'exports': []},
            {'type': 'code', 'code': 'result = 42', 'exports': ['result']},
            {'type': 'finish', 'answer': '完成任务'},
        ]
        for flags, visible in [(('--show-python',), True), (('--no-show-python',), False)]:
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as directory:
                terminal = Terminal(Path(directory), [actions], *flags)
                try:
                    terminal.send('执行并修正')
                    terminal.expect('完成任务')
                    terminal.expect('你> ')
                    self.assertNotIn('[debug:', terminal.text)
                    self.assertNotIn('host_rules', terminal.text)
                    self.assertEqual('```python' in terminal.text, visible)
                    self.assertEqual('[stdout · 执行 1]' in terminal.text, visible)
                    self.assertEqual('"result": 42' in terminal.text, visible)
                    if visible:
                        self.assertEqual(terminal.text.count("print('before-error')"), 1)
                        self.assertEqual(terminal.text.count('[stdout · 执行 1]'), 1)
                        self.assertIn('[Python · 执行 1 · 失败]', terminal.text)
                        self.assertIn('[Python · 执行 2 · 完成]', terminal.text)
                finally:
                    self.assertEqual(terminal.close(), 0)
