"""A model chooses code and its exports, or explicitly finishes the Turn."""

from dataclasses import dataclass
from typing import Protocol

from ..context import ModelContext


@dataclass(frozen=True)
class Code:
    code: str
    exports: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.code, str) or isinstance(self.exports, str):
            raise TypeError("Code requires source text and a sequence of export names")
        names = tuple(self.exports)
        if not all(isinstance(name, str) for name in names):
            raise TypeError("Export names must be strings")
        object.__setattr__(self, "exports", names)


@dataclass(frozen=True)
class Finish:
    answer: str

    def __post_init__(self):
        if not isinstance(self.answer, str):
            raise TypeError("Finish requires a text answer")


class Model(Protocol):
    def decide(self, context: ModelContext) -> Code | Finish:
        ...
