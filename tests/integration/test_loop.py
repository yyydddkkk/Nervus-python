from dataclasses import replace
from functools import partial
from pathlib import Path
import tempfile
import unittest

from nervus import (
    Capability, Code, Finish, ModelError, ScriptedModel, Session, WorkingStateLostError,
)
from tests.support.capabilities import Echo, SQLiteQuery


class LoopTests(unittest.TestCase):
    def test_model_selects_feedback_repairs_error_and_reuses_state_next_turn(self):
        def repair(context):
            self.assertEqual(context.input, "remember and compute")
            self.assertEqual(context.feedback[0].values, {"chosen": 42})
            self.assertIn("ZeroDivisionError", context.feedback[1].error)
            self.assertEqual(context.feedback[1].values, {})
            return Code("answer = add(2)", exports=("answer",))

        def finish(context):
            self.assertEqual(context.feedback[-1].values, {"answer": 43})
            return Finish("43")

        model = ScriptedModel([
            Code("saved = 40\nhidden = 'not exported'\ndef add(x): return saved + x\nchosen = add(2)",
                 exports=("chosen",)),
            Code("saved += 1\n1 / 0"),
            repair, finish,
        ])
        with Session() as session:
            result = session.run("remember and compute", model)
            self.assertEqual((result.reason, result.answer, result.decisions), ("finished", "43", 4))
            self.assertEqual([request.step for request in model.requests], [1, 2, 3, 4])
            self.assertTrue(all(request.turn == 1 for request in model.requests))
            for request in model.requests:
                self.assertTrue(all("hidden" not in feedback.values for feedback in request.feedback))
            second = ScriptedModel([Code("result = add(3)", exports=("result",)), Finish("44")])
            result = session.run("use the saved function", second)
            self.assertEqual(result.turn, 2)
            self.assertEqual(second.requests[1].feedback[0].values, {"result": 44})
            self.assertEqual(second.requests[0].feedback, ())

    def test_context_change_information_matches_active_and_next_snapshots(self):
        with tempfile.TemporaryDirectory() as directory, Session() as session:
            audit = str(Path(directory) / "audit.jsonl")
            old = Capability("read", partial(SQLiteQuery, audit, "old"), implementation_version="old")
            new = replace(old, factory=partial(SQLiteQuery, audit, "new"), implementation_version="new")
            session.publish([old])

            def host_update(context):
                self.assertEqual(context.revision, 1)
                self.assertEqual(context.capabilities[0].signature, "(value)")
                self.assertEqual(context.capability_changes[0].after.implementation_version, "old")
                session.publish([new])  # Host-controlled test action, not a model tool.
                return Code("reader = tools.read\nfirst = await reader('one')", exports=("first",))

            def continue_old(context):
                self.assertEqual(context.capabilities[0].implementation_version, "old")
                self.assertEqual(context.feedback[0].values["first"]["marker"], "old")
                return Code("again = await reader('two')", exports=("again",))

            first = ScriptedModel([host_update, continue_old, Finish("old turn complete")])
            session.run("query", first)
            self.assertTrue(all(request.revision == 1 for request in first.requests))

            def observe_new(context):
                self.assertEqual(context.revision, 2)
                change, = context.capability_changes
                self.assertEqual((change.before.implementation_version, change.after.implementation_version),
                                 ("old", "new"))
                return Code("current = await reader('three')", exports=("current",))

            second = ScriptedModel([observe_new, Finish("new turn complete")])
            session.run("query again", second)
            self.assertEqual(second.requests[1].feedback[0].values["current"]["marker"], "new")
            self.assertEqual([(c["revision"], c["implementation_version"]) for c in session.calls],
                             [(1, "old"), (1, "old"), (2, "new")])
            unchanged = ScriptedModel([Finish("nothing changed")])
            session.run("inspect", unchanged)
            self.assertEqual(unchanged.requests[0].capability_changes, ())
            session.publish([])
            removed = ScriptedModel([Finish("removed")])
            session.run("inspect removal", removed)
            self.assertEqual(removed.requests[0].capabilities, ())
            self.assertIsNone(removed.requests[0].capability_changes[0].after)

    def test_finish_and_decision_budget_both_drain_unfinished_tasks(self):
        code = Code("""import asyncio
effects = []
started = asyncio.Event()
gate = asyncio.Event()
async def late():
    started.set()
    await gate.wait()
    effects.append('late')
pending = asyncio.create_task(late())
await started.wait()
""")
        for max_decisions, reason in ((1, "decision_budget"), (3, "finished")):
            with self.subTest(reason=reason), Session() as session:
                model = ScriptedModel([code, Finish("done")])
                result = session.run("launch", model, max_decisions=max_decisions)
                self.assertEqual(result.reason, reason)
                self.assertEqual(result.tasks[0]["status"], "cancelled")
                self.assertEqual(len(model.requests), 1 if reason == "decision_budget" else 2)
                session.begin_turn()
                session.execute("assert pending.done() and pending.cancelled()\nassert effects == []")
                session.end_turn()

    def test_call_budget_stops_before_another_decision_and_does_not_overspend(self):
        with Session() as session:
            session.publish([Capability("echo", Echo)])
            model = ScriptedModel([
                Code("import asyncio\npending = asyncio.create_task(asyncio.Event().wait())\n"
                     "first = await tools.echo('ok')\nawait tools.echo('over budget')"),
                Finish("must not be requested"),
            ])
            result = session.run("two calls", model, call_budget=1)
            self.assertEqual(result.reason, "call_budget")
            self.assertEqual(len(model.requests), 1)
            self.assertIn("budget exhausted", result.feedback[0].error)
            self.assertEqual(result.tasks[0]["status"], "cancelled")
            self.assertEqual(len(session.calls), 1)
            self.assertEqual(session.calls[0]["result"], "ok")
            empty = ScriptedModel([Finish("must not be requested")])
            self.assertEqual(session.run("zero", empty, call_budget=0).reason, "call_budget")
            self.assertEqual(empty.requests, [])

    def test_model_contract_failure_still_closes_turn(self):
        with Session() as session:
            for model in (ScriptedModel([]), ScriptedModel([lambda context: {"invalid": True}])):
                with self.assertRaises(ModelError):
                    session.run("invalid script", model)
                # A failed model does not strand the Session in its old Turn.
                self.assertEqual(session.run("next", ScriptedModel([Finish("ok")])).answer, "ok")

    def test_working_state_loss_terminates_without_another_model_decision(self):
        with Session(execution_timeout=0.2) as session:
            model = ScriptedModel([Code("while True: pass"), Finish("must not be requested")])
            with self.assertRaises(WorkingStateLostError):
                session.run("uncooperative code", model)
            self.assertEqual(len(model.requests), 1)
            with self.assertRaises(WorkingStateLostError):
                session.run("do not rebuild", ScriptedModel([Finish("no")]))
