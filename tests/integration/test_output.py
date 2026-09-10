from dataclasses import replace
from functools import partial
from pathlib import Path
import tempfile
import unittest

from nervus import Capability, Code, Finish, ScriptedModel, Session
from tests.support.capabilities import SQLiteQuery


def text(output, channel=None, execution=None):
    return "".join(chunk["text"] for chunk in output["chunks"]
                   if (channel is None or chunk["channel"] == channel)
                   and (execution is None or chunk["execution"] == execution))


class OutputTests(unittest.TestCase):
    def test_capability_purpose_and_return_structure_follow_the_snapshot(self):
        with tempfile.TemporaryDirectory() as directory, Session() as session:
            audit = str(Path(directory) / "audit.jsonl")
            old = Capability("read", partial(SQLiteQuery, audit, "old"),
                             description="Query local data and increment its counter",
                             returns="dict: marker:str, count:int, value:input, instance:str")
            new = replace(old, factory=partial(SQLiteQuery, audit, "new"),
                          implementation_version="2", description="Query the replacement instance")
            session.publish([old])

            def publish(context):
                self.assertEqual(context.capabilities[0].description, old.description)
                self.assertEqual(context.capabilities[0].returns, old.returns)
                session.publish([new])
                return Code("row = await tools.read(4)", exports=("row",))

            def finish(context):
                self.assertEqual(context.capabilities[0].description, old.description)
                self.assertEqual(context.feedback[-1].values["row"]["marker"], "old")
                return Finish("old")

            session.run("read", ScriptedModel([publish, finish]))
            next_model = ScriptedModel([Finish("new")])
            session.run("inspect updated capability", next_model)
            self.assertEqual(next_model.requests[0].capabilities[0].description, new.description)
            self.assertEqual(next_model.requests[0].capabilities[0].returns, old.returns)

    def test_stdout_stderr_before_exception_reach_the_next_model_decision(self):
        def inspect_error(context):
            self.assertIn("diagnostic {'numbers': [2, 3]}", text(context.output, "stdout"))
            self.assertIn("warning", text(context.output, "stderr"))
            self.assertIn("ZeroDivisionError", context.feedback[-1].error)
            self.assertIn("<nervus-code turn=1 execution=1>:4", context.feedback[-1].error)
            self.assertEqual(context.feedback[-1].execution, 1)
            return Code("answer = 5", exports=("answer",))

        with Session() as session:
            result = session.run("diagnose", ScriptedModel([
                Code("import sys\nprint('diagnostic', {'numbers': [2, 3]})\n"
                     "print('warning', file=sys.stderr)\n1 / 0"),
                inspect_error, Finish("fixed"),
            ]))
            self.assertEqual(result.feedback[-1].values, {"answer": 5})
            self.assertIn("diagnostic", text(result.output))

    def test_output_is_bounded_across_entries_without_breaking_exports(self):
        def inspect_output(context):
            self.assertEqual(len(text(context.output)), 64)
            self.assertTrue(context.output["truncated"])
            self.assertGreater(context.output["dropped_characters"], 9900)
            self.assertEqual(context.feedback[-1].values, {"answer": 42})
            return Code("print('another entry')\nanswer += 1", exports=("answer",))

        with Session(output_limit=64) as session:
            result = session.run("bounded output", ScriptedModel([
                Code("import sys\nprint('x' * 10000)\nprint('stderr', file=sys.stderr)\nanswer = 42",
                     exports=("answer",)),
                inspect_output, Finish("43"),
            ]))
            self.assertEqual(len(text(result.output)), 64)
            self.assertEqual(result.feedback[-1].values, {"answer": 43})
            self.assertEqual(session.read_output(), result.output)
            session.begin_turn()
            session.execute("print('new')")
            self.assertEqual(text(session.read_output()), "new\n")
            self.assertFalse(session.read_output()["truncated"])
            session.end_turn()
        with Session() as session:
            session.begin_turn()
            session.execute("import sys\nfor _ in range(1000):\n sys.stdout.write('a'); sys.stderr.write('b')")
            output = session.read_output()
            self.assertLessEqual(len(output["chunks"]), 128)
            self.assertTrue(output["truncated"])
            session.end_turn()

    def test_background_and_cleanup_output_keep_their_creation_execution(self):
        first = Code("""import asyncio, sys
started = asyncio.Event()
gate = asyncio.Event()
progressed = asyncio.Event()
finish = asyncio.Event()
async def work():
    print('task started')
    started.set()
    await gate.wait()
    print('task continued')
    print('task warning', file=sys.stderr)
    progressed.set()
    await finish.wait()
    return 42
task = asyncio.create_task(work())
await started.wait()
""")

        def inspect_origin(context):
            self.assertIn("task continued", text(context.output, execution=1))
            self.assertIn("task warning", text(context.output, "stderr", execution=1))
            self.assertEqual(text(context.output, execution=2), "entry two\n")
            self.assertEqual([f.execution for f in context.feedback], [1, 2])
            return Code("""finish.set()
answer = await task
print('entry three')
cleanup_ready = asyncio.Event()
async def cleanup():
    cleanup_ready.set()
    try:
        await asyncio.Event().wait()
    finally:
        print('cleanup output')
pending = asyncio.create_task(cleanup())
await cleanup_ready.wait()
""", exports=("answer",))

        with Session() as session:
            result = session.run("background output", ScriptedModel([
                first, Code("gate.set()\nawait progressed.wait()\nprint('entry two')"),
                inspect_origin, Finish("42"),
            ]))
            self.assertEqual(result.feedback[-1].values, {"answer": 42})
            self.assertIn("cleanup output", text(result.output, execution=3))
            self.assertEqual([t["status"] for t in result.tasks], ["succeeded", "cancelled"])
            session.begin_turn()
            session.execute("assert task.done() and pending.cancelled()\nprint('next turn')")
            self.assertEqual(text(session.read_output()), "next turn\n")
            session.end_turn()
