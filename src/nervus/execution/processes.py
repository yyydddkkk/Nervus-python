"""Own accepted starts before Popen, independently of the Python worker's life."""

from copy import deepcopy
from dataclasses import dataclass, field
import os
import selectors
import signal
import subprocess
import threading
import time

from ..events import FIELDS
from ..errors import ProcessCleanupError, SessionStateError
from ..process import validate_argv


@dataclass
class _Start:
    fact: dict
    argv: list
    limit: int
    cancel: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    process: object = None
    error: str | None = None


class Processes:
    def __init__(self, journal):
        self.journal = journal
        self._lock = threading.RLock()
        self._starts = {}
        self._closed_turns = set()
        self._closed = False
        self._events = []

    def events(self):
        with self._lock:
            return tuple(deepcopy(self._events))

    def _record(self, start, event, **details):
        with self._lock:
            self._events.append({**start.fact, 'event': event, **details})

    def worker_exited(self, *, working_state_lost):
        with self._lock:
            self._events.append({'session': self.journal.session, 'generation': 1,
                                 'event': 'worker_exit_confirmed', 'working_state_lost': working_state_lost})

    def submit(self, fact, argv, limit, reply):
        argv = validate_argv(argv)
        if type(limit) is not int or limit < 0:
            raise ValueError('Invalid process output limit')
        with self._lock:
            if self._closed or fact['turn'] in self._closed_turns:
                raise SessionStateError('Process admission is closed for this Turn')
            admitted = next((c for c in self.journal.calls() if c['call'] == fact['call']), None)
            if (admitted is None or admitted['status'] != 'admitted'
                    or any(admitted[k] != fact[k] for k in FIELDS)
                    or fact['call'] in self._starts):
                raise SessionStateError('Process request has no matching admitted call')
            start = _Start({k: fact[k] for k in FIELDS}, argv, limit)
            self._starts[fact['call']] = start
            self._record(start, 'accepted')
        # The request is already owned; stop can mark it while creation blocks.
        threading.Thread(target=self._run, args=(start, reply), daemon=True).start()

    def cancel(self, call):
        with self._lock:
            start = self._starts.get(call)
            if start is not None and not start.done.is_set():
                start.cancel.set()

    def close_admission(self, turn=None):
        with self._lock:
            if turn is None:
                self._closed = True
            else:
                self._closed_turns.add(turn)
            for start in self._starts.values():
                if (turn is None or start.fact['turn'] == turn) and not start.done.is_set():
                    start.cancel.set()

    def wait(self, turn=None, timeout=2):
        deadline = time.monotonic() + timeout
        with self._lock:
            starts = [s for s in self._starts.values() if turn is None or s.fact['turn'] == turn]
        for start in starts:
            if not start.done.wait(max(0, deadline - time.monotonic())) or start.error:
                raise ProcessCleanupError(
                    f"External process cleanup unconfirmed for call {start.fact['call']}; "
                    "the Turn is not confirmed stopped")

    def check_begin(self):
        with self._lock:
            if any(not s.done.is_set() or s.error for s in self._starts.values()):
                raise ProcessCleanupError('Previous external execution is not confirmed stopped')

    def _spawn(self, argv):
        return subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, start_new_session=True, bufsize=0)

    @staticmethod
    def _signal(process, sig):
        # Never signal a group after reaping its leader: the PGID could be reused.
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass

    def _collect(self, start):
        process = start.process
        data = {'stdout': bytearray(), 'stderr': bytearray()}
        truncated = {'stdout': False, 'stderr': False}
        terminate_at = None
        killed = False
        with selectors.DefaultSelector() as selector:
            for name in data:
                stream = getattr(process, name)
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, name)
            while True:
                if start.cancel.is_set() and terminate_at is None:
                    self._record(start, 'stopping', pid=process.pid)
                    self._signal(process, signal.SIGTERM)
                    terminate_at = time.monotonic()
                # WNOWAIT pins the leader PID through the final group signal.
                exited = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                if not killed and (exited is not None or
                                   (terminate_at is not None and time.monotonic() - terminate_at >= 0.1)):
                    if terminate_at is None:
                        self._record(start, 'stopping', pid=process.pid, reason='leader_exited')
                    self._signal(process, signal.SIGKILL)
                    killed = True
                for key, _ in selector.select(0.01):
                    chunk = os.read(key.fileobj.fileno(), 8192)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                        continue
                    name = key.data
                    remaining = start.limit - len(data[name])
                    data[name].extend(chunk[:remaining])
                    truncated[name] |= len(chunk) > remaining
                if exited is not None and not selector.get_map():
                    break
                if terminate_at is not None and time.monotonic() - terminate_at > 1.5:
                    raise ProcessCleanupError('External process or its output pipes did not stop')
            code = process.wait(timeout=1)
        # Group disappearance is deliberately conservative, including zombies.
        deadline = time.monotonic() + 1
        while True:
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                break
            if time.monotonic() >= deadline:
                raise ProcessCleanupError('External process group exit unconfirmed')
            time.sleep(0.01)
        return {'returncode': code,
                **{name: bytes(value).decode('utf-8', errors='replace') for name, value in data.items()},
                **{name + '_truncated': flag for name, flag in truncated.items()}}

    def _run(self, start, reply):
        result = None
        failure = None
        try:
            self._record(start, 'starting')
            start.process = self._spawn(start.argv)
            self._record(start, 'running', pid=start.process.pid)
            result = self._collect(start)
            self._record(start, 'exited', pid=start.process.pid, returncode=result['returncode'],
                         stop_requested=start.cancel.is_set())
        except Exception as error:
            failure = f'{type(error).__name__}: {error}'
            if start.process is None:
                self._record(start, 'start_failed', error=failure)
            else:
                # Monitoring failure never abandons a still-owned OS resource.
                # Do not signal after wait() has reaped the group leader.
                try:
                    if start.process.returncode is None:
                        self._signal(start.process, signal.SIGKILL)
                        start.process.wait(timeout=1)
                except Exception as cleanup_error:
                    failure += f'; cleanup: {type(cleanup_error).__name__}: {cleanup_error}'
                finally:
                    for stream in (start.process.stdout, start.process.stderr):
                        stream.close()
                start.error = failure
                self._record(start, 'cleanup_unconfirmed', pid=start.process.pid, error=failure)
        finally:
            start.done.set()
        reply({'call': start.fact['call'], 'ok': failure is None, 'result': result,
               'error': failure, 'cancelled': start.cancel.is_set()})
