"""Model-visible context from the selected Turn, not the current registry."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CapabilityView:
    name: str
    identity: str
    interface_version: str
    implementation_version: str
    signature: str
    description: str = ""
    returns: str = ""


@dataclass(frozen=True)
class CapabilityChange:
    name: str
    before: CapabilityView | None
    after: CapabilityView | None


@dataclass(frozen=True)
class Feedback:
    step: int
    code: str
    exports: tuple[str, ...]
    values: dict[str, Any]
    error: str | None = None
    execution: int | None = None


@dataclass(frozen=True)
class ModelContext:
    input: str
    turn: int
    revision: int
    step: int
    remaining_calls: int
    capabilities: tuple[CapabilityView, ...]
    capability_changes: tuple[CapabilityChange, ...]
    feedback: tuple[Feedback, ...]
    output: dict[str, Any] = field(default_factory=dict)


def capability_changes(previous, current):
    before = {cap.name: cap for cap in previous}
    after = {cap.name: cap for cap in current}
    return tuple(CapabilityChange(name, before.get(name), after.get(name))
                 for name in sorted(before.keys() | after.keys())
                 if before.get(name) != after.get(name))
