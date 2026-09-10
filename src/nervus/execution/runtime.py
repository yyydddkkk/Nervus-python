"""Own process transport and exit confirmation, hidden from the Host interface."""

import multiprocessing
import threading

from .worker import run_worker
from .processes import Processes
from .. import errors
from ..events import Journal


class Runtime:
    def __init__(self, session: str, timeout: float, drain_timeout: float, output_limit: int):
        self.timeout = timeout
        self.journal = Journal(session)
        self.processes = Processes(self.journal)
        self._turn = None
        self._process_reply_lock = threading.Lock()
        self._commands_lock = threading.RLock()
        self._exit_lock = threading.RLock()
        self._joined = False
        self._closed = False
        self._lost = False
        context = multiprocessing.get_context("spawn")
        self._commands, child_commands = context.Pipe()
        self._facts, child_facts = context.Pipe()
        self._stops, child_stops = context.Pipe()
        self._process_requests, child_process_requests = context.Pipe()
        self._process = context.Process(target=run_worker,
                                        args=(child_commands, child_facts, session, drain_timeout, output_limit, child_stops, child_process_requests))
        try:
            self._process.start()
        except Exception:
            self._commands.close()
            self._facts.close()
            self._stops.close()
            self._process_requests.close()
            raise
        finally:
            child_commands.close()
            child_facts.close()
            child_stops.close()
            child_process_requests.close()
        self._receiver = threading.Thread(target=self._receive_facts, daemon=True)
        self._receiver.start()
        self._process_receiver = threading.Thread(target=self._receive_processes, daemon=True)
        self._process_receiver.start()

    def _reply_process(self, response):
        try:
            with self._process_reply_lock:
                self._process_requests.send(response)
        except (EOFError, OSError):
            pass  # Resource ownership and exit facts survive an undeliverable result.

    def _receive_processes(self):
        try:
            while True:
                request = self._process_requests.recv()
                if request["op"] == "cancel":
                    self.processes.cancel(request["call"])
                    continue
                try:
                    self.processes.submit(request["fact"], request["argv"], request["limit"], self._reply_process)
                except Exception as error:
                    self._reply_process({"call": request["fact"]["call"], "ok": False,
                                         "cancelled": False, "error": f"{type(error).__name__}: {error}"})
        except (EOFError, OSError):
            self.processes.close_admission()

    def _receive_facts(self):
        try:
            while True:
                fact = self._facts.recv()
                self._facts.send(self.journal.accept(fact))
        except (EOFError, OSError):
            return  # Only the command owner joins the process, avoiding waitpid races.

    def _join(self, terminate: bool):
        with self._exit_lock:
            if not self._joined:
                self._join_once(terminate)
                self._joined = True

    def _join_once(self, terminate: bool):
        self.processes.close_admission()
        if terminate and self._process.is_alive():
            self._process.terminate()
        def confirm():
            self._process.join(2)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(2)
            if self._process.is_alive():
                raise errors.SessionStateError("Worker exit unconfirmed; Session cannot continue")

        self.journal.confirm_exit(confirm)
        self.processes.worker_exited(working_state_lost=terminate)
        self._receiver.join(2)
        if self._receiver.is_alive():
            raise errors.SessionStateError("Call fact receiver did not stop")
        self._commands.close()
        self._facts.close()
        self._stops.close()
        self._process_receiver.join(2)
        self._process_requests.close()

    def _lose_worker(self, reason):
        with self._exit_lock:
            self._join(terminate=True)
            self._lost = True
            self.processes.wait()
        raise errors.WorkingStateLostError(
            f"{reason}. Worker exit confirmed; all variables, functions and capability state lost. "
            "Confirmed calls remain available; unfinished calls are unknown. No replay was attempted."
        )

    def stop(self, turn, grace):
        # Admission closes before either worker cancellation or OS creation completes.
        if turn != self._turn:
            raise errors.SessionStateError('Target Turn is not current')
        self.processes.close_admission(turn)
        try:
            self._stops.send(turn)
            if not self._stops.poll(grace):
                self._lose_worker("Turn stop exceeded its grace period")
            response = self._stops.recv()
        except (EOFError, OSError):
            self._lose_worker("Worker stop communication ended")
        if not response["ok"]:
            if response["error"] == "TurnDrainError":
                self._lose_worker(response["message"])
            raise getattr(errors, response["error"], errors.NervusError)(response["message"])
        self.processes.wait(turn)
        return response["result"]

    def request(self, command, *, check=None, **arguments):
        with self._commands_lock:
            if self._closed:
                raise errors.SessionClosedError("Session is closed")
            if self._lost:
                raise errors.WorkingStateLostError("This Session's working environment was lost")
            if check is not None:
                check()
            if command == "begin":
                self.processes.check_begin()
            if command == "end" and arguments.get("target_turn", self._turn) != self._turn:
                raise errors.SessionStateError('Target Turn is not current')
            if command in {"end", "close"}:
                self.processes.close_admission(self._turn if command == "end" else None)
            try:
                self._commands.send((command, arguments))
                if not self._commands.poll(self.timeout):
                    self._lose_worker(f"{command} exceeded its deadline")
                response = self._commands.recv()
            except (EOFError, OSError):
                self._lose_worker("Worker communication ended")
            if not response["ok"] and response["error"] == "TurnDrainError":
                self._lose_worker(response["message"])
            if command in {"end", "close"}:
                self.processes.wait(self._turn if command == "end" else None)
            if not response["ok"]:
                error_type = getattr(errors, response["error"], errors.NervusError)
                raise error_type(response["message"])
            if command == "begin":
                self._turn = response["result"]["turn"]
            return response["result"]

    def close(self):
        with self._commands_lock:
            if self._closed:
                self.processes.wait()  # A previous cleanup failure is not erased by close().
                return
            try:
                if not self._lost:
                    self.request("close")
            finally:
                self._closed = True
                self._join(terminate=self._lost)
                self.processes.wait()
