"""Run with: uv run python examples/basic_session.py"""

from dataclasses import replace
from functools import partial

from nervus import Capability, Session


class TaggedEcho:
    def __init__(self, marker):
        self.marker = marker

    async def initialize(self):
        self.calls = 0

    async def __call__(self, value):
        self.calls += 1
        return {"marker": self.marker, "value": value, "calls": self.calls}

    async def close(self):
        pass


def main():
    old = Capability("read", partial(TaggedEcho, "old"), implementation_version="old")
    new = replace(old, factory=partial(TaggedEcho, "new"), implementation_version="new")
    with Session() as session:
        session.publish([old])
        session.begin_turn()
        session.execute("saved = [20, 22]\ndef total(): return sum(saved)\nreader = tools.read")
        print("Turn 1:", session.execute("before = await reader(total())", exports=["before"]))
        session.publish([new])
        print("After publish, Turn 1:", session.execute("during = await reader(total())", exports=["during"]))
        session.end_turn()
        session.begin_turn()
        print("Turn 2:", session.execute("after = await reader(total())", exports=["after", "saved"]))
        session.end_turn()


if __name__ == "__main__":
    main()
