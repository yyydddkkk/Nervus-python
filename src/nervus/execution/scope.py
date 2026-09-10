"""Execution origin belongs to a Turn, not to whichever entry runs next."""

import asyncio
from contextvars import ContextVar, copy_context
from dataclasses import dataclass, field

from ..composition.environment import Snapshot
from ..errors import SessionStateError, TurnDrainError
from .output import OutputBuffer


origin = ContextVar("nervus_origin", default=None)


@dataclass(eq=False)
class TurnScope:
    number: int
    snapshot: Snapshot
    remaining_calls: int
    closing: bool = False
    tasks: dict = field(default_factory=dict)
    output: OutputBuffer = field(default_factory=OutputBuffer)

    def check(self):
        if self.closing:
            raise SessionStateError("Turn is closing or ended")

    async def drain(self, timeout: float) -> list[dict]:
        self.closing = True
        for task in self.tasks:
            if not task.done():
                task.cancel()
        deadline = asyncio.get_running_loop().time() + timeout
        while pending := {task for task in self.tasks if not task.done()}:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TurnDrainError("Turn tasks did not stop within the grace period")
            await asyncio.wait(pending, timeout=remaining)
        for task, record in self.tasks.items():
            collect(task, record)
        report = [dict(record) for record in self.tasks.values()]
        self.tasks.clear()
        return report


def collect(task, record):
    if "status" in record:
        return
    if task.cancelled():
        record["status"] = "cancelled"
    else:
        error = task.exception()
        record["status"] = "failed" if error is not None else "succeeded"
        if error is not None:
            record["error"] = f"{type(error).__name__}: {error}"


def task_factory(loop, coroutine, **kwargs):
    source = origin.get()
    context = kwargs.pop("context", None)
    context = copy_context() if context is None else context.copy()
    context.run(origin.set, source)
    kwargs["eager_start"] = False
    task = asyncio.Task(coroutine, loop=loop, context=context, **kwargs)
    if source is not None:
        scope, execution = source
        record = {"task": len(scope.tasks) + 1, "turn": scope.number, "execution": execution}
        scope.tasks[task] = record
        task.add_done_callback(lambda done: collect(done, record))
        if scope.closing:
            task.cancel()
    return task
