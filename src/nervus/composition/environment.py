"""Prepare complete revisions; release only instances no snapshot still needs."""

from dataclasses import dataclass
from types import MappingProxyType
from collections.abc import Mapping, Sequence

from .capability import Capability, CapabilityInstance
from ..errors import CapabilityInitializationError, CapabilityReleaseError


@dataclass(eq=False)
class Binding:
    capability: Capability
    instance: CapabilityInstance
    release_attempted: bool = False


@dataclass(frozen=True)
class Snapshot:
    revision: int
    names: Mapping[str, str]
    bindings: Mapping[str, Binding]


class Environment:
    def __init__(self):
        self.published = Snapshot(0, MappingProxyType({}), MappingProxyType({}))
        self._instances: list[Binding] = []

    async def _release(self, instances: Sequence[Binding]) -> list[str]:
        failures = []
        for binding in reversed(instances):
            if binding.release_attempted:
                continue
            binding.release_attempted = True
            try:
                await binding.instance.close()
            except Exception as error:
                failures.append(f"{binding.capability.name}: {type(error).__name__}: {error}")
        return failures

    async def publish(self, capabilities: tuple[Capability, ...], retained: Snapshot | None) -> int:
        return await self._publish_plan(capabilities, retained)

    async def update(self, add, replace, remove, retained: Snapshot | None) -> int:
        current = self.published.bindings
        targets = set(remove) | replace.keys()
        if len(remove) != len(set(remove)) or set(remove) & replace.keys():
            raise CapabilityInitializationError("Duplicate or overlapping update targets")
        if targets - current.keys():
            raise CapabilityInitializationError("Update target identity is not published")
        if any(cap.identity in current for cap in add):
            raise CapabilityInitializationError("Existing identities require explicit replacement")
        if not add and not replace and not remove:
            return self.published.revision
        # Only explicit targets change. Unmentioned entries are the actual old
        # Bindings, not descriptions to reconstruct or factories to compare.
        plan = [replace.get(identity, binding) for identity, binding in current.items()
                if identity not in remove]
        plan.extend(add)
        return await self._publish_plan(plan, retained)

    async def _publish_plan(self, plan, retained: Snapshot | None) -> int:
        capabilities = [item.capability if isinstance(item, Binding) else item for item in plan]
        if (len({c.name for c in capabilities}) != len(capabilities)
                or len({c.identity for c in capabilities}) != len(capabilities)):
            raise CapabilityInitializationError("Duplicate capability name or identity")
        created = []
        bindings = []
        try:
            for item in plan:
                if isinstance(item, Binding):
                    bindings.append(item)
                    continue
                # Factories must construct resource-free instances; initialize
                # acquires resources so partial failures can always be closed.
                instance = item.factory()
                if any(instance is prior.instance for prior in (*self._instances, *created)):
                    raise TypeError("Each factory must return a fresh capability instance")
                binding = Binding(item, instance)
                created.append(binding)
                bindings.append(binding)
                await instance.initialize()
        except Exception as error:
            failures = await self._release(created)
            raise CapabilityInitializationError(
                f"Candidate not published: {type(error).__name__}: {error}; cleanup failures: {failures}"
            ) from error
        self._instances.extend(created)
        self.published = Snapshot(
            self.published.revision + 1,
            MappingProxyType({c.name: c.identity for c in capabilities}),
            MappingProxyType({b.capability.identity: b for b in bindings}),
        )
        await self.retire(retained)
        return self.published.revision

    async def retire(self, retained: Snapshot | None) -> None:
        keep = set(self.published.bindings.values())
        if retained is not None:
            keep.update(retained.bindings.values())
        retired = [binding for binding in self._instances if binding not in keep]
        failures = await self._release(retired)
        self._instances = [binding for binding in self._instances if binding in keep]
        if failures:
            raise CapabilityReleaseError(
                f"Revision {self.published.revision} remains published; release failures: {failures}"
            )

    async def close(self) -> None:
        failures = await self._release(self._instances)
        self._instances.clear()
        if failures:
            raise CapabilityReleaseError(f"Session resource release failures: {failures}")
