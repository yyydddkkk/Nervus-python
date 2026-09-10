"""Supervisor-owned, in-memory acceptance of call facts; not object recovery."""

from copy import deepcopy
import threading


FIELDS = ("session", "generation", "turn", "execution", "revision", "call",
          "identity", "interface_version", "implementation_version")


class Journal:
    def __init__(self, session: str):
        self.session = session
        self._calls = {}
        self._lock = threading.RLock()
        self._lost = False

    def accept(self, fact: dict) -> bool:
        with self._lock:
            if (self._lost or any(key not in fact for key in FIELDS)
                    or fact["session"] != self.session or fact["generation"] != 1):
                return False
            key = fact["call"]
            prior = self._calls.get(key)
            if fact["status"] == "admitted":
                if prior is not None:
                    return False
            elif fact["status"] in {"succeeded", "failed", "cancelled"}:
                if (prior is None or prior["status"] != "admitted"
                        or any(prior[k] != fact[k] for k in FIELDS)):
                    return False
            else:
                return False
            self._calls[key] = deepcopy(fact)
            return True

    def confirm_exit(self, confirm):
        """Order physical exit confirmation and sealing against fact acceptance."""
        with self._lock:
            confirm()  # Must raise if exit cannot be confirmed.
            self.worker_lost()

    def worker_lost(self):
        with self._lock:
            self._lost = True
            for record in self._calls.values():
                if record["status"] == "admitted":
                    record.update(status="interrupted", outcome="unknown")

    def calls(self) -> tuple[dict, ...]:
        with self._lock:
            return tuple(deepcopy(list(self._calls.values())))
