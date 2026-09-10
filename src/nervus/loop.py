"""One Host Input becomes one bounded Turn, with explicit selected feedback."""

from copy import deepcopy
from dataclasses import dataclass, field

from .context import CapabilityView, Feedback, ModelContext, capability_changes
from .errors import CodeExecutionError, ModelError, WorkingStateLostError, TurnStoppedError
from .models.model import Code, Finish, Model


@dataclass(frozen=True)
class TurnResult:
    turn: int
    revision: int
    reason: str
    answer: str | None
    decisions: int
    feedback: tuple[Feedback, ...]
    tasks: tuple[dict, ...]
    output: dict = field(default_factory=dict)


def run_turn(session, input: str, model: Model, max_decisions: int, call_budget: int, owner) -> TurnResult:
    if not isinstance(input, str):
        raise TypeError("Input must be text")
    if type(max_decisions) is not int or max_decisions < 1:
        raise ValueError("max_decisions must be a positive integer")
    if type(call_budget) is not int or call_budget < 0:
        raise ValueError("call_budget must be a nonnegative integer")
    turn = session._turn_request("begin", owner=owner, call_budget=call_budget)["turn"]

    def request(command, **arguments):
        return session._turn_request(command, owner=owner, target_turn=turn, **arguments)

    lost = False
    stopped = False
    view = {"turn": turn, "revision": None}
    feedback = []
    decisions = 0
    answer = None
    reason = "decision_budget"
    try:
        view = request("describe")
        capabilities = tuple(CapabilityView(**cap) for cap in view["capabilities"])
        changes = capability_changes(session._model_capabilities, capabilities)
        while decisions < max_decisions:
            # Derived from the active worker scope; background tasks may also
            # consume calls. The invocation path is still the final budget gate.
            remaining = request("describe")["remaining_calls"]
            if remaining == 0:
                reason = "call_budget"
                break
            context = ModelContext(input, view["turn"], view["revision"], decisions + 1,
                                   remaining, capabilities, changes, tuple(feedback), request("output"))
            with session._control_lock:
                session._check_run(owner)
                session._model_capabilities = capabilities  # Last view actually presented.
            decisions += 1  # A started request counts even if its response becomes stale.
            try:
                action = model.decide(deepcopy(context))
            except Exception:
                session._check_run(owner)  # A late model exception cannot revive control.
                raise
            session._check_run(owner)
            if isinstance(action, Finish):
                reason, answer = "finished", action.answer
                break
            if not isinstance(action, Code):
                raise ModelError("Model must return Code or Finish")
            try:
                values = request("execute", code=action.code, exports=action.exports)
                execution = request("output")["last_execution"]
                feedback.append(Feedback(decisions, action.code, action.exports, values, execution=execution))
            except CodeExecutionError as error:
                execution = request("output")["last_execution"]
                feedback.append(Feedback(decisions, action.code, action.exports, {}, str(error), execution))
            if request("describe")["remaining_calls"] == 0:
                reason = "call_budget"
                break
    except TurnStoppedError:
        stopped = True
    except WorkingStateLostError:
        lost = True
        raise  # No model retry, namespace replay or automatic reconstruction.
    finally:
        if not lost and not stopped:
            try:
                tasks = request("end")  # Also drain on model/protocol exceptions.
                output = request("output")
            except TurnStoppedError:
                stopped = True
        if stopped:
            report = session._stop_result(owner)
            tasks, output = report["tasks"], report["output"]
            reason, answer = "stopped", None
            view["revision"] = report["revision"]
    return TurnResult(view["turn"], view["revision"], reason, answer, decisions,
                      tuple(feedback), tuple(tasks), output)
