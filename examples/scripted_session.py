"""Run with: uv run python examples/scripted_session.py (no model API)."""

from dataclasses import replace
from functools import partial

from nervus import Capability, Code, Finish, ScriptedModel, Session
from basic_session import TaggedEcho


def correct(context):
    assert "NameError" in context.feedback[-1].error
    return Code("result = await reader(total())", exports=("result",))


def finish(context):
    value = context.feedback[-1].values["result"]
    return Finish(f"{value['marker']}: {value['value']}")


def main():
    old = Capability("read", partial(TaggedEcho, "old"), implementation_version="old")
    new = replace(old, factory=partial(TaggedEcho, "new"), implementation_version="new")
    with Session() as session:
        session.publish([old])
        first = ScriptedModel([
            Code("saved = [20, 22]\ndef total(): return sum(saved)\n"
                 "reader = tools.read\nresult = await reader(missing)", exports=("result",)),
            correct, finish,
        ])
        print(session.run("Compute and keep the working state", first).answer)
        session.publish([new])
        second = ScriptedModel([Code("result = await reader(total())", exports=("result",)), finish])
        print(session.run("Use the saved state with the updated capability", second).answer)
        print("Capability change:", second.requests[0].capability_changes)


if __name__ == "__main__":
    main()
