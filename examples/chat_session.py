"""Interactive Host: uv run --locked python examples/chat_session.py."""

import argparse
from dataclasses import asdict
from functools import partial
import json
import os
from pathlib import Path
import queue
import re
import signal
import threading

from nervus import Capability, Code, ProcessRunner, Session
from nervus.errors import SessionStateError
from nervus.models.deepseek import DeepSeekFlash


ROOT = Path(__file__).resolve().parents[1]


def load_env(path):
    """Small literal KEY=value subset; no shell execution or interpolation."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('export '):
            line = line[7:].strip()
        key, sep, value = line.partition('=')
        key, value = key.strip(), value.strip()
        if not sep or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key):
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


class Job:
    """Daemon ownership: a blocked synchronous decide must not hold exit open."""

    def __init__(self, session, model, text, events, decisions, calls):
        self.cancelled = False
        self.model = model
        self.session = session
        self.events = events
        self.decisions = decisions
        self.calls = calls
        self.thread = threading.Thread(target=self.run, args=(text,), daemon=True)

    def emit(self, kind, value):
        self.events.put((self, kind, value))

    def decide(self, context):
        self.emit('feedback', {'feedback': [asdict(f) for f in context.feedback],
                               'output': context.output})
        action = self.model.decide(context)
        # The UI also checks job identity: no late response becomes live output.
        if not self.cancelled and isinstance(action, Code):
            self.emit('code', action.code)
        return action

    def run(self, text):
        try:
            result = self.session.run(text, self, max_decisions=self.decisions,
                                      call_budget=self.calls)
            self.emit('result', result)
        except Exception as error:
            self.emit('error', type(error).__name__)

    def stop(self):
        self.cancelled = True
        turn = self.session.active_turn
        if turn is None:
            return False  # Begin may still be in flight. Retry only this job.
        try:
            report = self.session.stop(turn, grace=1)
        except SessionStateError:
            if self.session.active_turn is None:
                return False  # The captured Turn ended normally before stop.
            raise
        self.stop_report = report
        return True


def read_input(events):
    while True:
        try:
            line = input()
        except EOFError:
            events.put((None, 'quit', None))
            return
        events.put((None, 'input', line))
        if line.strip() == '/quit':
            return


def show(label, value):
    if label == 'code':
        print('[code]\n' + value, flush=True)
        return
    print(f'[{label}]', json.dumps(value, ensure_ascii=False, default=str), flush=True)


def positive(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-decisions', type=positive, default=8)
    parser.add_argument('--call-budget', type=positive, default=20)
    parser.add_argument('--max-tokens', type=positive, default=1536)
    args = parser.parse_args(argv)
    load_env(ROOT / '.env')
    if not os.environ.get('DEEPSEEK_API_KEY', '').strip():
        parser.error('Set DEEPSEEK_API_KEY in the environment or project .env')
    if os.environ.get('DEEPSEEK_MODEL', 'deepseek-v4-flash') != 'deepseek-v4-flash':
        parser.error('Only deepseek-v4-flash is supported by this adapter')
    events = queue.Queue()
    interrupted = False

    def on_interrupt(signum, frame):
        # No locks in a signal handler: even Event.set() can be interrupted
        # while the main thread owns that same Event's lock.
        nonlocal interrupted
        interrupted = True

    previous_handler = signal.signal(signal.SIGINT, on_interrupt)
    exit_code = 0
    session = None
    job = None
    quitting = False
    try:
        session = Session(execution_timeout=10, drain_timeout=1, output_limit=4096)
        session.publish([Capability('run', partial(ProcessRunner, output_limit=4096),
            description='Run argv without shell parsing and wait for exit. Runs on the host; not a sandbox.',
            returns='dict: returncode, stdout, stderr, stdout_truncated, stderr_truncated')])
        print('Nervus /help /workspace /quit | Ctrl+C stops the current Turn.\n'
              'WARNING: generated Python and programs run on this host, without approval or sandbox.\n'
              'State persists; conversation history does not. Enter a task:', flush=True)
        threading.Thread(target=read_input, args=(events,), daemon=True).start()
        while True:
            try:
                if interrupted:
                    interrupted = False
                    if job:
                        job.cancelled = True
                        print('Stopping current Turn; synchronous model HTTP cannot be forcibly cancelled.', flush=True)
                    else:
                        print('No active Turn. /quit to exit.', flush=True)
                if job is not None and job.cancelled:
                    if job.stop():
                        show('stop', job.stop_report)
                        show('usage-so-far', [asdict(r) for r in getattr(job.model, 'records', ())])
                        show('stopped', 'Worker work drained. Synchronous model request may still run and incur cost; late response ignored.')
                        job = None
                    elif not job.thread.is_alive():
                        job = None
                if quitting and job is None:
                    break
                try:
                    sender, kind, value = events.get(timeout=0.05)
                except queue.Empty:
                    continue
                if sender is not None:
                    if sender is not job or sender.cancelled:
                        continue
                    if kind == 'result':
                        show('feedback', [asdict(f) for f in value.feedback])
                        show('output', value.output)
                        show('answer', value.answer)
                        show('turn', {'reason': value.reason, 'decisions': value.decisions,
                                      'capability_calls': len([c for c in session.calls if c['turn'] == value.turn])})
                        show('usage', [asdict(r) for r in getattr(job.model, 'records', ())])
                        job = None
                        print('Enter a task:', flush=True)
                    elif kind == 'error':
                        show('error', value)  # Do not expose arbitrary exception text / credentials.
                        show('usage', [asdict(r) for r in getattr(job.model, 'records', ())])
                        job = None
                        if value == 'WorkingStateLostError':
                            quitting = True
                    else:
                        show(kind, value)
                    continue
                text = value.strip() if kind == 'input' else '/quit'
                if text == '/quit':
                    quitting = True
                    if job:
                        job.cancelled = True
                elif text == '/help':
                    print('One line = one Turn. /workspace shows metadata when idle. /quit or EOF closes.\n'
                          'During work: Ctrl+C stops; /quit exits. Other tasks are rejected, not queued.\n'
                          'Use named variables across Turns; no chat history or automatic recovery.', flush=True)
                elif job:
                    print('Busy. Ctrl+C to stop, or /quit to exit.', flush=True)
                elif text == '/workspace':
                    show('workspace', session.inspect_namespace())
                elif text.startswith('/'):
                    print('Unknown command. /help', flush=True)
                elif text:
                    model = DeepSeekFlash(base_url=os.environ.get('DEEPSEEK_BASE_URL', 'https://api.deepseek.com'),
                                          timeout=30, max_tokens=args.max_tokens)
                    job = Job(session, model, text, events, args.max_decisions, args.call_budget)
                    job.thread.start()
            except KeyboardInterrupt:
                if job:
                    job.cancelled = True
                    print('Stopping current Turn; synchronous model HTTP cannot be forcibly cancelled.', flush=True)
                else:
                    print('No active Turn. /quit to exit.', flush=True)
    except Exception as error:
        show('fatal', type(error).__name__)
        exit_code = 1
    finally:
        try:
            if session is not None:
                session.close()
        except Exception as error:
            show('cleanup-error', type(error).__name__)
            exit_code = 1
        else:
            if session is not None:
                print('Session closed.', flush=True)
        finally:
            signal.signal(signal.SIGINT, previous_handler)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
