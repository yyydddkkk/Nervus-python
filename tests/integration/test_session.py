from dataclasses import replace
from functools import partial
import json
from pathlib import Path
import tempfile
import unittest

from nervus import (
    Session, Capability, CodeExecutionError, CapabilityInitializationError,
    CapabilityReleaseError, SessionClosedError, WorkingStateLostError,
)
from tests.support.capabilities import SQLiteQuery, Echo


class SessionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.audit = str(Path(directory.name) / "lifecycle.jsonl")

    def capability(self, marker, **options):
        return Capability("read", partial(SQLiteQuery, self.audit, marker, **options),
                          implementation_version=marker)

    def records(self, phase=None):
        entries = [json.loads(line) for line in Path(self.audit).read_text().splitlines()]
        return entries if phase is None else [e for e in entries if e["phase"] == phase]

    def test_host_path_persists_state_updates_snapshot_and_releases_real_resources(self):
        old = self.capability("old")
        new = replace(old, factory=partial(SQLiteQuery, self.audit, "new"), implementation_version="new")
        with Session() as session:
            self.assertEqual(session.publish([old]), 1)
            self.assertEqual(session.begin_turn(), {"turn": 1, "revision": 1})
            values = session.execute("""saved = [20, 22]
def total(): return sum(saved)
reader = tools.read
def capture(reference):
    async def call(value): return await reference(value)
    return call
closed_reader = capture(reader)
first = await reader(total())
""", exports=["first"])
            self.assertEqual(values["first"]["marker"], "old")
            self.assertEqual(session.publish([new]), 2)
            self.assertEqual([e["marker"] for e in self.records("ready")], ["old", "new"])
            self.assertEqual(self.records("closed"), [])
            old_result = session.execute("during = await tools.read(total())", exports=["during"])
            self.assertEqual(old_result["during"]["marker"], "old")
            self.assertEqual(old_result["during"]["count"], 2)
            session.end_turn()
            self.assertEqual([e["marker"] for e in self.records("closed")], ["old"])
            self.assertEqual(session.begin_turn(), {"turn": 2, "revision": 2})
            result = session.execute("after = await closed_reader(total())", exports=["after", "saved"])
            self.assertEqual(result["saved"], [20, 22])
            self.assertEqual((result["after"]["marker"], result["after"]["count"]), ("new", 1))
            self.assertNotEqual(result["after"]["instance"], values["first"]["instance"])
            session.end_turn()
        session.close()  # Host close is idempotent, without a second resource close.
        self.assertEqual([e["marker"] for e in self.records("closed")], ["old", "new"])
        self.assertTrue(all(e["verified"] and e["close_count"] == 1 for e in self.records("closed")))
        self.assertEqual([(c["turn"], c["revision"], c["implementation_version"], c["status"])
                          for c in session.calls],
                         [(1, 1, "old", "succeeded"), (1, 1, "old", "succeeded"),
                          (2, 2, "new", "succeeded")])
        with self.assertRaises(SessionClosedError):
            session.begin_turn()

    def test_partial_initialization_failure_cleans_candidates_and_keeps_old_environment(self):
        old = self.capability("old")
        with Session() as session:
            session.publish([old])
            session.begin_turn()
            session.execute("reader = tools.read")
            first = replace(old, factory=partial(SQLiteQuery, self.audit, "candidate"))
            failing = replace(self.capability("failing", fail_initialize=True), name="other")
            with self.assertRaisesRegex(CapabilityInitializationError, "Candidate not published"):
                session.publish([first, failing])
            self.assertEqual([e["marker"] for e in self.records("closed")], ["failing", "candidate"])
            self.assertTrue(all(e["verified"] for e in self.records("closed")))
            self.assertEqual(session.execute("result = await reader(1)", exports=["result"])["result"]["marker"], "old")
            session.end_turn()
            self.assertEqual(session.begin_turn()["revision"], 1)
            session.execute("result = await reader(2)")
            session.end_turn()
        self.assertEqual([e["marker"] for e in self.records("closed")], ["failing", "candidate", "old"])

    def test_aliases_follow_identity_and_interface_rules_through_public_entry(self):
        cap = Capability("echo", Echo)
        with Session() as session:
            session.publish([cap])
            session.begin_turn()
            session.execute("alias = tools.echo")
            session.end_turn()
            session.publish([Capability("echo", Echo)])  # Same name, different identity.
            session.begin_turn()
            with self.assertRaisesRegex(CodeExecutionError, "unavailable"):
                session.execute("await alias('old')")
            session.end_turn()
            session.publish([cap])  # Explicit restoration.
            session.begin_turn()
            self.assertEqual(session.execute("result = await alias('restored')", exports=["result"]),
                             {"result": "restored"})
            session.end_turn()
            session.publish([replace(cap, interface_version="2")])
            session.begin_turn()
            with self.assertRaisesRegex(CodeExecutionError, "interface version mismatch"):
                session.execute("await alias('incompatible')")
            session.end_turn()

    def test_tasks_continue_across_entries_and_close_drains_before_releasing(self):
        with Session() as session:
            session.publish([self.capability("task")])
            session.begin_turn()
            session.execute("""import asyncio
gate = asyncio.Event()
started = asyncio.Event()
async def run():
    started.set()
    await gate.wait()
    return await tools.read('finished')
task = asyncio.create_task(run())
await started.wait()
""")
            result = session.execute("gate.set()\nvalue = await task", exports=["value"])
            self.assertEqual(result["value"]["marker"], "task")
            session.execute("pending = asyncio.create_task(asyncio.Event().wait())")
            report = session.end_turn()
            self.assertEqual([r["status"] for r in report], ["succeeded", "cancelled"])
            session.begin_turn()
            session.execute("assert pending.done() and pending.cancelled()")
            with self.assertRaisesRegex(CodeExecutionError, "CancelledError"):
                session.execute("await pending")
            session.execute("assert task.done()")
            # Closing an active Turn must drain and release as well.
        self.assertEqual(len(self.records("closed")), 1)
        self.assertTrue(self.records("closed")[0]["verified"])
        self.assertEqual(session.calls[0]["execution"], 1)

    def test_failed_entry_keeps_state_and_sessions_have_independent_namespaces(self):
        with Session() as first, Session() as second:
            first.begin_turn()
            second.begin_turn()
            with self.assertRaises(CodeExecutionError):
                first.execute("saved = 42\nraise ValueError('entry failed')")
            self.assertEqual(first.execute("copied = saved", exports=["copied"]), {"copied": 42})
            second.execute("assert 'saved' not in globals()")
            first.end_turn()
            second.end_turn()

    def test_public_capability_name_does_not_resolve_to_worker_internals(self):
        with Session() as session:
            session.publish([Capability("worker", Echo)])
            session.begin_turn()
            self.assertEqual(session.execute("value = await tools.worker('ordinary capability')",
                                             exports=["value"]),
                             {"value": "ordinary capability"})
            session.end_turn()

    def test_release_failure_is_reported_and_other_resources_still_close(self):
        with self.assertRaisesRegex(CapabilityReleaseError, "release failure"):
            with Session() as session:
                session.publish([self.capability("bad", fail_close=True),
                                 replace(self.capability("good"), name="other")])
        self.assertEqual({e["marker"] for e in self.records("closed")}, {"bad", "good"})
        self.assertTrue(all(e["verified"] for e in self.records("closed")))
        session.close()

    def test_uncooperative_execution_reports_state_loss_without_replay(self):
        with Session(execution_timeout=0.3) as session:
            session.publish([Capability("echo", Echo)])
            session.begin_turn()
            session.execute("result = await tools.echo('confirmed')")
            with self.assertRaisesRegex(WorkingStateLostError, "Worker exit confirmed"):
                session.execute("while True: pass")
            self.assertEqual(session.calls[0]["result"], "confirmed")
            with self.assertRaises(WorkingStateLostError):
                session.begin_turn()
