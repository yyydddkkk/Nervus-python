"""A small synchronous Host interface; delegate model loops and worker transport."""

from collections.abc import Iterable, Mapping
from uuid import uuid4
import threading
import math
from copy import deepcopy
from dataclasses import dataclass, field

from .composition.capability import Capability
from .execution.runtime import Runtime
from .execution.namespace import validate_limits
from .loop import run_turn, TurnResult
from .models.model import Model
from .errors import SessionStateError, SessionClosedError, TurnStoppedError, WorkingStateLostError


@dataclass(eq=False)
class _RunControl:
    turn: int | None = None
    stopping: bool = False
    done: threading.Event = field(default_factory=threading.Event)
    report: dict | None = None
    error: Exception | None = None


class Session:
    def __init__(self, *, execution_timeout: float = 10, drain_timeout: float = 1, output_limit: int = 4096):
        if execution_timeout <= 0 or drain_timeout <= 0:
            raise ValueError("Timeouts must be positive")
        if type(output_limit) is not int or output_limit < 0:
            raise ValueError("output_limit must be a nonnegative character count")
        self.identity = uuid4().hex
        self._runtime = Runtime(self.identity, execution_timeout, drain_timeout, output_limit)
        self._model_capabilities = ()
        self._control_lock = threading.RLock()
        self._run_owner = None
        self._closing = False

    def _check_run(self, owner):
        with self._control_lock:
            if owner is not None and owner.stopping:
                raise TurnStoppedError("Turn stop requested")
            if self._closing:
                raise SessionClosedError("Session is closing or closed")
            if self._run_owner is not owner:
                raise SessionStateError("Turn is controlled by Session.run()")

    def _turn_request(self, command, *, owner=None, **arguments):
        self._check_run(owner)
        try:
            result = self._runtime.request(command, check=lambda: self._check_run(owner), **arguments)
        except Exception:
            self._check_run(owner)  # Stop wins over a late command response/error.
            raise
        if command == "begin" and owner is not None:
            with self._control_lock:
                owner.turn = result["turn"]
        return result

    @property
    def active_turn(self) -> int | None:
        """The Turn owned by run(), including stopping; None outside a model run."""
        with self._control_lock:
            return self._run_owner.turn if self._run_owner is not None else None

    def stop(self, turn: int, *, grace: float = 1) -> dict:
        """Stop a specific model Turn; wait for drain or raise state loss after exit.

        A synchronous model request may remain blocked. Its eventual response is
        discarded, and it does not retain control after stopping completes.
        """
        if type(turn) is not int or turn < 1 or not math.isfinite(grace) or grace <= 0:
            raise ValueError("A positive Turn number and grace period are required")
        with self._control_lock:
            owner = self._run_owner
            if owner is None or owner.turn != turn or owner.stopping:
                raise SessionStateError("Target model Turn is not active or is already stopping")
            owner.stopping = True
        try:
            owner.report = self._runtime.stop(turn, grace)
            return deepcopy(owner.report)
        except Exception as error:
            owner.error = error
            raise
        finally:
            with self._control_lock:
                if owner.report is not None or isinstance(owner.error, WorkingStateLostError):
                    if self._run_owner is owner:
                        self._run_owner = None
                owner.done.set()

    def _stop_result(self, owner):
        owner.done.wait()
        if owner.error is not None:
            raise owner.error
        return deepcopy(owner.report)

    def publish(self, capabilities: Iterable[Capability]) -> int:
        """Prepare a full replacement; an active Turn retains its snapshot."""
        values = tuple(capabilities)
        if not all(isinstance(value, Capability) for value in values):
            raise TypeError("publish expects Capability descriptions")
        return self._runtime.request("publish", capabilities=values)

    def update(self, *, add: Iterable[Capability] = (),
               replace: Mapping[str, Capability] | None = None,
               remove: Iterable[str] = ()) -> int:
        """Apply explicit identity-targeted changes; preserve untouched instances."""
        if replace is not None and not isinstance(replace, Mapping):
            raise TypeError("replace must map existing identities to Capability descriptions")
        if isinstance(remove, str):
            raise TypeError("remove must be a sequence of identities, not one string")
        additions = tuple(add)
        replacements = dict(replace) if replace is not None else {}
        removals = tuple(remove)
        if (not all(isinstance(c, Capability) for c in (*additions, *replacements.values()))
                or not all(isinstance(identity, str) for identity in (*replacements, *removals))):
            raise TypeError("update expects Capability values and identity strings")
        return self._runtime.request("update", add=additions, replace=replacements, remove=removals)

    def begin_turn(self, *, call_budget: int = 100) -> dict:
        if type(call_budget) is not int or call_budget < 0:
            raise ValueError("call_budget must be a nonnegative integer")
        return self._turn_request("begin", call_budget=call_budget)

    def execute(self, code: str, *, exports: Iterable[str] = ()) -> dict:
        """Execute in the persistent namespace and return selected plain-data values."""
        names = tuple(exports)
        if not isinstance(code, str) or not all(isinstance(name, str) for name in names):
            raise TypeError("code and export names must be strings")
        return self._turn_request("execute", code=code, exports=names)

    def end_turn(self) -> list[dict]:
        """Cancel/drain remaining tasks, then release retired capability instances."""
        return self._turn_request("end")

    def describe_turn(self) -> dict:
        """Describe the selected snapshot and its remaining budget, not the registry."""
        return self._turn_request("describe")

    def read_output(self) -> dict | None:
        """Bounded current/last Turn output, labeled by its originating execution."""
        return self._turn_request("output")

    def inspect_namespace(self, *, max_entries=50, max_bytes=8192, prefix="") -> dict:
        """On-demand, bounded metadata; no repr, contents, or persistent index."""
        validate_limits(max_entries, max_bytes, prefix)
        return self._turn_request("inspect_namespace", max_entries=max_entries,
                                     max_bytes=max_bytes, prefix=prefix)

    def run(self, input: str, model: Model, *, max_decisions: int = 8,
            call_budget: int = 100) -> TurnResult:
        """Accept one Input and finish/drain one Turn before returning."""
        with self._control_lock:
            if self._closing:
                raise SessionClosedError("Session is closing or closed")
            if self._run_owner is not None:
                raise SessionStateError("Turn is controlled by Session.run()")
            owner = self._run_owner = _RunControl()
        try:
            return run_turn(self, input, model, max_decisions, call_budget, owner)
        finally:
            with self._control_lock:
                if self._run_owner is owner and not owner.stopping:
                    self._run_owner = None

    @property
    def calls(self) -> tuple[dict, ...]:
        """Copies of supervisor-confirmed call records, readable after close/loss."""
        return self._runtime.journal.calls()

    @property
    def inspections(self) -> tuple[dict, ...]:
        """Completed model workspace inspections; read while idle, not model context."""
        return tuple(self._turn_request("inspections"))

    @property
    def process_events(self) -> tuple[dict, ...]:
        """Supervisor-owned external process facts, separate from call outcomes."""
        return self._runtime.processes.events()

    def close(self) -> None:
        """Drain an active Turn, release resources and confirm process exit; idempotent."""
        with self._control_lock:
            owner = self._run_owner
            if owner is not None and not (owner.stopping and owner.done.is_set() and owner.error is not None):
                raise SessionStateError("Turn is controlled by Session.run()")
            self._closing = True
        self._runtime.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
