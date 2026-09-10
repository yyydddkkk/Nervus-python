"""Synthetic execution and diagnostic regressions for workspace discovery.

These scripted decisions test the kernel, not autonomous model performance.
"""

from dataclasses import asdict, replace
from functools import partial
import json
import unittest
from unittest.mock import patch
from uuid import uuid4

from nervus import Capability, Code, Finish, ScriptedModel, Session


# Standalone task inputs. No provider transcripts or internal evidence are needed.
PREPARATION = "Read the numbers and retain their list, a zero-argument total function, and the reader."
REUSE = "Discover the saved objects, reuse them, and add the old total to newly fetched numbers."
DESCRIPTION = "Return a synthetic number packet from the current implementation."
RETURNS = "A mapping with numbers, marker, and instance fields."


class Numbers:
    def __init__(self, marker, values):
        self.marker, self.values = marker, values

    async def initialize(self):
        self.instance = uuid4().hex

    async def __call__(self, token=None):
        return {"numbers": list(self.values), "marker": self.marker, "instance": self.instance}

    async def close(self):
        pass


SETUP = Code("""packet = await tools.read_numbers()
notebook_values = packet['numbers']
def notebook_total(): return sum(notebook_values)
notebook_reader = tools.read_numbers
before_ids = {'data': id(notebook_values), 'function': id(notebook_total), 'reference': id(notebook_reader)}
initial_total = notebook_total()
""", exports=("initial_total", "before_ids"))


class DiscoveryRegressionTests(unittest.TestCase):
    def setUp(self):
        blocked = patch("urllib.request.urlopen", side_effect=AssertionError("Deterministic regression attempted network"))
        blocked.start()
        self.addCleanup(blocked.stop)

    def prepare(self, session):
        old = Capability("read_numbers", partial(Numbers, "old", [19, 37, 61]), implementation_version="old",
                         description=DESCRIPTION, returns=RETURNS)
        session.publish([old])
        result = session.run(PREPARATION, ScriptedModel([SETUP, Finish("ready")]),
                             max_decisions=5, call_budget=10)
        self.assertEqual(result.feedback[0].values["initial_total"], 117)
        session.publish([replace(old, factory=partial(Numbers, "new", [101, 103]), implementation_version="new")])
        return result.feedback[0].values["before_ids"]

    def choose(self, catalog):
        self.assertFalse(catalog["truncated"])
        data = next(e for e in catalog["entries"] if e["kind"] == "value" and e["type"] == "list")
        function = next(e for e in catalog["entries"] if e["kind"] == "function")
        reference = next(e for e in catalog["entries"] if e["kind"] == "capability_reference")
        self.assertEqual(function["signature"], "()")
        self.assertTrue(reference["callable"])
        return data["name"], function["name"], reference["name"]

    def result_code(self, names, fetch):
        data, function, reference = names
        prefix = f"fresh = await {reference}()\n" if fetch else ""
        return Code(prefix + f"old_total = {function}()\nnew_total = sum(fresh['numbers'])\n"
                    "total = old_total + new_total\nmarker = fresh['marker']\n"
                    f"after_ids = {{'data': id({data}), 'function': id({function}), 'reference': id({reference})}}",
                    exports=("old_total", "new_total", "total", "marker", "after_ids"))

    def verify_result(self, result, model, before_ids, session, decisions):
        self.assertEqual((result.reason, result.decisions), ("finished", decisions))
        values = result.feedback[-1].values
        self.assertEqual((values["old_total"], values["new_total"], values["total"], values["marker"]),
                         (117, 204, 321, "new"))
        self.assertEqual(values["after_ids"], before_ids)
        self.assertEqual(model.requests[0].feedback, ())
        self.assertEqual(model.requests[0].output["chunks"], [])
        self.assertNotIn("notebook_", json.dumps(asdict(model.requests[0])))
        self.assertEqual([(c["turn"], c["implementation_version"], c["status"]) for c in session.calls],
                         [(1, "old", "succeeded"), (2, "new", "succeeded")])
        session.begin_turn()  # The loop really closed Turn 2.
        session.end_turn()

    def test_correct_discovery_and_reuse_finishes_within_the_original_budget(self):
        with Session() as session:
            before_ids = self.prepare(session)

            def compute(context):
                names = self.choose(context.feedback[-1].values["catalog"])
                return self.result_code(names, fetch=True)

            model = ScriptedModel([Code("catalog = inspect_workspace()", exports=("catalog",)),
                                   compute, Finish("117 + 204 = 321; new")])
            result = session.run(REUSE, model, max_decisions=5, call_budget=10)
            self.verify_result(result, model, before_ids, session, decisions=3)

    def test_original_misuses_have_actionable_feedback_and_can_be_corrected(self):
        with Session() as session:
            before_ids = self.prepare(session)
            chosen = []

            def correct_import(context):
                error = context.feedback[-1].error
                self.assertIn("ModuleNotFoundError: No module named 'inspect_workspace'", error)
                self.assertIn("<nervus-code turn=2 execution=2>:1", error)
                return Code("catalog = inspect_workspace()", exports=("catalog",))

            def wrong_arguments(context):
                names = self.choose(context.feedback[-1].values["catalog"])
                chosen[:] = names
                _, function, reference = names
                return Code(f"fresh = await {reference}()\nprint('function signature: ()')\n"
                            f"wrong = {function}(fresh['numbers'])")

            def correct_arguments(context):
                error = context.feedback[-1].error
                self.assertIn("TypeError: notebook_total() takes 0 positional arguments but 1 was given", error)
                self.assertIn("<nervus-code turn=2 execution=4>:3", error)
                self.assertIn("function signature: ()", "".join(c["text"] for c in context.output["chunks"]))
                return self.result_code(chosen, fetch=False)  # No unnecessary capability retry.

            model = ScriptedModel([Code("import inspect_workspace\ncatalog = inspect_workspace()"),
                                   correct_import, wrong_arguments, correct_arguments, Finish("321; new")])
            result = session.run(REUSE, model, max_decisions=5, call_budget=10)
            self.verify_result(result, model, before_ids, session, decisions=5)
