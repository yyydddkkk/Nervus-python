"""Each step is an action or a callback that checks feedback before choosing one."""

from collections.abc import Callable, Iterable
from copy import deepcopy

from .model import Code, Finish
from ..context import ModelContext
from ..errors import ModelError


class ScriptedModel:
    def __init__(self, steps: Iterable[Code | Finish | Callable[[ModelContext], Code | Finish]]):
        self._steps = iter(steps)
        self.requests: list[ModelContext] = []

    def decide(self, context: ModelContext) -> Code | Finish:
        self.requests.append(deepcopy(context))
        try:
            step = next(self._steps)
        except StopIteration as error:
            raise ModelError("Script exhausted before Finish or a budget boundary") from error
        return step(context) if callable(step) else step
