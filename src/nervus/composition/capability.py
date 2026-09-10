"""The Host supplies construction; ordinary instances own worker-local resources."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import uuid4


class CapabilityInstance(Protocol):
    async def initialize(self) -> None:
        """Acquire resources. close() must also work after partial initialization."""
        ...

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Return plain data; leave no unowned background execution."""
        ...

    async def close(self) -> None:
        """Release resources; the kernel makes at most one attempt per instance."""
        ...


@dataclass(frozen=True)
class Capability:
    name: str
    factory: Callable[[], CapabilityInstance]
    identity: str = field(default_factory=lambda: uuid4().hex)
    interface_version: str = "1"
    implementation_version: str = "1"
    description: str = ""
    returns: str = ""

    def __post_init__(self):
        if not self.name.isidentifier() or self.name.startswith("_"):
            raise ValueError("Capability names must be public Python identifiers")
        if not all((self.identity, self.interface_version, self.implementation_version)):
            raise ValueError("Identity and versions must be nonempty")
        if not callable(self.factory):
            raise TypeError("factory must be callable and spawn-serializable")
        if not isinstance(self.description, str) or not isinstance(self.returns, str):
            raise TypeError("description and returns must be text")


def copy_data(value: Any) -> Any:
    """Clone the first slice's explicit data contract, without exporting handles."""
    if type(value) in (type(None), bool, int, float, str, bytes):
        return value
    if type(value) in (list, tuple):
        return type(value)(copy_data(item) for item in value)
    if type(value) is dict and all(type(key) is str for key in value):
        return {key: copy_data(item) for key, item in value.items()}
    raise TypeError("Capability values and exports must be plain data, not live objects")
