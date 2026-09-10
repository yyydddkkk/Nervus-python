from dataclasses import asdict, replace
import json
import unittest

from nervus import Capability, Code, Finish, ScriptedModel, Session
from tests.support.capabilities import Echo


def entries(directory):
    return {entry["name"]: entry for entry in directory["entries"]}


def encoded_size(directory):
    return len(json.dumps(directory, ensure_ascii=True, separators=(",", ":")).encode("ascii"))


class NamespaceTests(unittest.TestCase):
    def test_live_directory_survives_turns_and_tracks_delete_and_rename(self):
        with Session() as session:
            self.assertEqual(session.inspect_namespace(), {"entries": [], "truncated": False})
            session.begin_turn()
            session.execute("values = [2, 3, 5]\n_private = 7\n"
                            "def compute(x, y=8, /, *args, flag=True, **kwargs): return x + y")
            directory = entries(session.inspect_namespace())
            self.assertEqual(set(directory), {"values", "_private", "compute"})
            self.assertEqual(directory["values"]["length"], 3)
            self.assertEqual(directory["compute"]["signature"], "(x, y=…, /, *args, flag=…, **kwargs)")
            session.end_turn()
            self.assertEqual(entries(session.inspect_namespace()), directory)
            session.begin_turn()
            session.execute("renamed = values\ndel values\nalias = compute\ndel compute\ndel _private")
            current = entries(session.inspect_namespace())
            self.assertEqual(set(current), {"renamed", "alias"})
            self.assertEqual(current["renamed"]["length"], 3)
            self.assertEqual(current["alias"]["signature"], directory["compute"]["signature"])
            session.end_turn()

    def test_directory_never_invokes_object_hooks_or_leaks_contents_and_defaults(self):
        with Session() as session:
            session.begin_turn()
            session.execute("""hits = []
class Meta(type):
    @property
    def __name__(cls):
        hits.append('metaclass property')
        raise AssertionError('unsafe')
    def __getattribute__(cls, name):
        hits.append('metaclass lookup')
        return type.__getattribute__(cls, name)
    def __hash__(cls):
        hits.append('metaclass hash')
        return type.__hash__(cls)
class Hostile(metaclass=Meta):
    def __getattribute__(self, name):
        hits.append('instance lookup')
        return object.__getattribute__(self, name)
    def __repr__(self):
        hits.append('repr')
        return 'TOP_SECRET_PAYLOAD'
    def __len__(self):
        hits.append('len')
        return 999
    @property
    def __signature__(self):
        hits.append('signature property')
        return None
class HostileList(list):
    def __len__(self):
        hits.append('subclass len')
        return 999
class HostileDict(dict):
    def keys(self):
        hits.append('subclass keys')
        return super().keys()
def annotation():
    hits.append('lazy annotation')
    return int
obj = Hostile()
subclass = HostileList([1, 2])
huge = 'TOP_SECRET_PAYLOAD' * 100000
mapping = {'SECRET_KEY': 'SECRET_VALUE'}
def function(value: annotation() = obj, *, option='DEFAULT_SECRET'): pass
function.__signature__ = obj
function.__wrapped__ = obj
function.__kwdefaults__ = HostileDict(option=obj)
hits.clear()
""")
            directory = session.inspect_namespace(max_entries=100)
            rendered = json.dumps(directory)
            for content in ("TOP_SECRET_PAYLOAD", "SECRET_KEY", "SECRET_VALUE", "DEFAULT_SECRET"):
                self.assertNotIn(content, rendered)
            values = entries(directory)
            self.assertEqual(values["obj"]["type"], "Hostile")
            self.assertNotIn("length", values["obj"])
            self.assertNotIn("length", values["subclass"])
            self.assertEqual(values["huge"]["length"], len("TOP_SECRET_PAYLOAD") * 100000)
            self.assertEqual(values["mapping"]["length"], 1)
            self.assertIn("value=…", values["function"]["signature"])
            self.assertEqual(session.execute("unchanged = hits", exports=["unchanged"]), {"unchanged": []})
            session.end_turn()

    def test_entry_field_and_serialized_output_limits(self):
        with Session(output_limit=80) as session:
            with self.assertRaises(ValueError):
                session.inspect_namespace(max_entries=101)
            with self.assertRaises(ValueError):
                session.inspect_namespace(max_bytes=16385)
            session.begin_turn()
            session.execute("for i in range(200): globals()[f'item_{i:03d}'] = 'hidden' * 1000\n"
                            "globals()['long_' + 'x' * 1000] = 1")
            directory = session.inspect_namespace(prefix="item_", max_entries=3)
            self.assertEqual(len(directory["entries"]), 3)
            self.assertTrue(directory["truncated"])
            limited = session.inspect_namespace(prefix="item_", max_entries=100, max_bytes=256)
            self.assertLessEqual(encoded_size(limited), 256)
            self.assertTrue(limited["truncated"])
            long_name = session.inspect_namespace(prefix="long_")
            self.assertTrue(long_name["entries"][0]["metadata_truncated"])
            self.assertLessEqual(len(long_name["entries"][0]["name"]), 97)
            result = session.execute("catalog = inspect_workspace(prefix='item_', max_bytes=256)\nprint(catalog)",
                                     exports=["catalog"])
            self.assertLessEqual(encoded_size(result["catalog"]), 256)
            self.assertTrue(session.read_output()["truncated"])
            session.end_turn()

    def test_reference_presence_is_not_current_turn_availability(self):
        cap = Capability("echo", Echo)
        with Session() as session:
            session.publish([cap])
            session.begin_turn()
            session.execute("reader = tools.echo")

            def reader():
                return entries(session.inspect_namespace(prefix="reader"))["reader"]

            self.assertEqual(reader()["kind"], "capability_reference")
            self.assertTrue(reader()["callable"])
            session.publish([])
            self.assertTrue(reader()["callable"])  # Active Turn retains its old snapshot.
            session.end_turn()
            self.assertEqual(reader()["availability"], "no_active_turn")
            session.begin_turn()
            self.assertFalse(reader()["callable"])
            self.assertEqual(reader()["availability"], "unavailable")
            session.end_turn()
            session.publish([Capability("echo", Echo)])
            session.begin_turn()
            self.assertFalse(reader()["callable"])  # Same name never substitutes identity.
            session.end_turn()
            session.publish([replace(cap, interface_version="2")])
            session.begin_turn()
            self.assertEqual(reader()["availability"], "interface_mismatch")
            session.end_turn()
            session.publish([cap])
            session.begin_turn(call_budget=0)
            self.assertEqual(reader()["availability"], "budget_exhausted")
            session.end_turn()
            self.assertEqual(session.calls, ())  # Inspection never invoked the capability.

    def test_model_requests_directory_explicitly_and_no_catalog_is_auto_injected(self):
        with Session() as session:
            session.begin_turn()
            session.execute("retained_name = [17, 19]\ndef retained_function(): return sum(retained_name)")
            session.end_turn()

            def inspect_catalog(context):
                values = context.feedback[-1].values["catalog"]
                self.assertEqual(entries(values)["retained_name"]["length"], 2)
                self.assertEqual(entries(values)["retained_function"]["kind"], "function")
                return Code("answer = retained_function()", exports=("answer",))

            model = ScriptedModel([
                Code("catalog = inspect_workspace(prefix='retained_')", exports=("catalog",)),
                inspect_catalog, Finish("36"),
            ])
            result = session.run("Discover and use the working environment", model)
            self.assertNotIn("retained_name", json.dumps(asdict(model.requests[0])))
            self.assertEqual(result.feedback[-1].values, {"answer": 36})
            later = ScriptedModel([Finish("No inspection requested")])
            session.run("finish", later)
            self.assertNotIn("retained_name", json.dumps(asdict(later.requests[0])))
