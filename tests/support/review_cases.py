"""Deterministic barriers and resource probes retained from the Kernel review.

Only the late-fact test uses private transport hooks to order process exit and
fact acceptance; all outcomes are observed through Session.
"""

from dataclasses import replace
from functools import partial
import json
from pathlib import Path
import tempfile
import threading

from nervus import Capability, Session
from tests.support.capabilities import Echo, SQLiteQuery


def close_after_retirement_failure(*, fail_new=False):
    with tempfile.TemporaryDirectory() as directory:
        audit = str(Path(directory) / "resources.jsonl")
        session = Session()
        cap = Capability("read", partial(SQLiteQuery, audit, "old", fail_close=True))
        session.publish([cap])
        session.begin_turn()
        session.publish([replace(cap, factory=partial(SQLiteQuery, audit, "new", fail_close=fail_new), name="replacement", implementation_version="new")])
        try:
            session.close()
            error = None
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            session.close()
        events = [json.loads(line) for line in Path(audit).read_text().splitlines()]
        return {"error": error, "ready": [e["marker"] for e in events if e["phase"] == "ready"],
                "close_hooks_called": [e["marker"] for e in events if e["phase"] == "closed"],
                "closed": [e for e in events if e["phase"] == "closed"]}


def late_terminal_after_confirmed_exit():
    arrived, allow_terminal, exited = threading.Event(), threading.Event(), threading.Event()
    observed = {"order": []}
    with Session(execution_timeout=0.5) as session:
        session.publish([Capability("echo", Echo)])
        session.begin_turn()
        runtime = session._runtime
        accept = runtime.journal.accept
        join = runtime._process.join
        lost = runtime.journal.worker_lost

        def delayed_accept(fact):
            if fact["status"] == "succeeded":
                arrived.set()
                if not allow_terminal.wait(3):
                    raise RuntimeError("Probe terminal barrier timed out")
            accepted = accept(fact)
            if fact["status"] == "succeeded":
                observed["order"].append("terminal_accepted" if accepted else "terminal_rejected")
            return accepted

        def confirmed_join(timeout=None):
            join(timeout)
            if runtime._process.exitcode is not None and not exited.is_set():
                observed["order"].append("process_exit_confirmed")
                exited.set()

        def record_lost():
            lost()
            observed["order"].append("journal_worker_lost")

        runtime.journal.accept = delayed_accept
        runtime._process.join = confirmed_join
        runtime.journal.worker_lost = record_lost

        def execute():
            try:
                session.execute("result = await tools.echo('computed')")
            except Exception as exc:
                observed["error"] = f"{type(exc).__name__}: {exc}"

        runner = threading.Thread(target=execute, daemon=True)
        runner.start()
        try:
            assert arrived.wait(2)
            assert exited.wait(2)
        finally:
            allow_terminal.set()
            runner.join(3)
        assert not runner.is_alive()
        observed["calls"] = session.calls
    return observed


def falsey_task_exception():
    with Session() as session:
        session.begin_turn()
        session.execute("""import asyncio
class QuietFailure(Exception):
    def __bool__(self): return False
ready = asyncio.Event()
async def fail():
    try:
        raise QuietFailure('background failure')
    finally:
        ready.set()
pending = asyncio.create_task(fail())
await ready.wait()
""")
        report = session.end_turn()
        session.begin_turn()
        try:
            session.execute("await pending")
            actual_error = None
        except Exception as exc:
            actual_error = f"{type(exc).__name__}: {exc}"
        session.end_turn()
        return {"task_report": report, "await_error": actual_error}

