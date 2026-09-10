from dataclasses import replace
from functools import partial
import json
from pathlib import Path
import tempfile
import unittest

from nervus import Capability, CapabilityInitializationError, CapabilityReleaseError, CodeExecutionError, Session
from tests.support.capabilities import SQLiteQuery


class PartialUpdateTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.audit = str(Path(directory.name) / 'resources.jsonl')

    def cap(self, name, marker, **options):
        return Capability(name, partial(SQLiteQuery, self.audit, marker, **options), implementation_version=marker)

    def records(self, phase):
        return [r for line in Path(self.audit).read_text().splitlines() if (r := json.loads(line))['phase'] == phase]

    def test_shared_instance_keeps_connection_state_while_active_turn_pins_replaced_instance(self):
        kept, old = self.cap('cache', 'kept'), self.cap('read', 'old')
        new = replace(old, factory=partial(SQLiteQuery, self.audit, 'new'), implementation_version='new')
        with Session() as session:
            session.publish([kept, old])
            session.begin_turn()
            first = session.execute("reader = tools.read\na = await tools.cache(0)\nb = await reader(0)", exports=['a','b'])
            self.assertEqual(session.update(replace={old.identity: new}), 2)
            during = session.execute('a = await tools.cache(1)\nb = await reader(1)', exports=['a','b'])
            self.assertEqual(during['a']['instance'], first['a']['instance'])
            self.assertEqual(during['a']['count'], 2)
            self.assertEqual(during['b']['marker'], 'old')
            self.assertEqual(self.records('closed'), [])
            session.end_turn()
            self.assertEqual([r['marker'] for r in self.records('closed')], ['old'])
            session.begin_turn()
            after = session.execute('a = await tools.cache(2)\nb = await reader(2)', exports=['a','b'])
            self.assertEqual(after['a']['instance'], first['a']['instance'])
            self.assertEqual(after['a']['count'], 3)
            self.assertEqual((after['b']['marker'], after['b']['count']), ('new', 1))
            session.end_turn()
        self.assertEqual([r['marker'] for r in self.records('ready')].count('kept'), 1)
        closed = self.records('closed')
        self.assertCountEqual([r['marker'] for r in closed], ['old','kept','new'])
        self.assertTrue(all(r['verified'] and r['close_count'] == 1 for r in closed))

    def test_prepare_failure_only_cleans_new_instances_even_when_cleanup_fails(self):
        kept, old = self.cap('cache','kept'), self.cap('read','old')
        candidate = replace(old, factory=partial(SQLiteQuery,self.audit,'candidate',fail_close=True))
        failing = self.cap('extra','failing', fail_initialize=True, fail_close=True)
        with Session() as session:
            session.publish([kept,old])
            session.begin_turn()
            first = session.execute('a = await tools.cache(1)', exports=['a'])['a']
            with self.assertRaisesRegex(CapabilityInitializationError, 'cleanup failures'):
                session.update(replace={old.identity:candidate}, add=[failing])
            self.assertCountEqual([r['marker'] for r in self.records('closed')], ['candidate','failing'])
            session.end_turn()
            self.assertEqual(session.begin_turn()['revision'], 1)
            values = session.execute('a = await tools.cache(2)\nb = await tools.read(2)', exports=['a','b'])
            self.assertEqual((values['a']['instance'], values['a']['count']), (first['instance'],2))
            self.assertEqual(values['b']['marker'],'old')
            session.end_turn()
        self.assertTrue(all(r['verified'] and r['close_count']==1 for r in self.records('closed')))

    def test_remove_restore_and_incompatible_replacement_follow_reference_rules(self):
        original = self.cap('read','original')
        with Session() as session:
            session.publish([original])
            session.begin_turn()
            session.execute('alias = tools.read')
            session.end_turn()
            session.update(remove=[original.identity])
            session.begin_turn()
            with self.assertRaisesRegex(CodeExecutionError,'unavailable'):
                session.execute('await alias(1)')
            session.end_turn()
            session.update(add=[original])
            session.begin_turn()
            session.execute('await alias(2)')
            session.end_turn()
            session.update(replace={original.identity:replace(original,interface_version='2')})
            session.begin_turn()
            with self.assertRaisesRegex(CodeExecutionError,'interface version'):
                session.execute('await alias(3)')
            session.end_turn()

    def test_retirement_failure_does_not_skip_other_resources_or_undo_publication(self):
        kept, bad, good = self.cap('cache','kept'), self.cap('bad','bad',fail_close=True), self.cap('good','good')
        with Session() as session:
            session.publish([kept,bad,good])
            with self.assertRaises(CapabilityReleaseError):
                session.update(remove=[bad.identity,good.identity])
            self.assertCountEqual([r['marker'] for r in self.records('closed')], ['bad','good'])
            self.assertEqual(session.begin_turn()['revision'],2)
            self.assertEqual([c['name'] for c in session.describe_turn()['capabilities']],['cache'])
            session.execute('await tools.cache(1)')
            session.end_turn()
        self.assertTrue(all(r['verified'] and r['close_count']==1 for r in self.records('closed')))

    def test_invalid_plans_do_not_initialize_and_empty_update_does_not_publish(self):
        cap = self.cap('read','original')
        with Session() as session:
            session.publish([cap])
            for arguments in ({'remove':['missing']}, {'replace':{'missing':cap}},
                              {'remove':[cap.identity,cap.identity]},
                              {'replace':{cap.identity:cap},'remove':[cap.identity]},
                              {'add':[cap]}, {'add':[self.cap('read','collision')]}):
                with self.subTest(arguments=arguments), self.assertRaises(CapabilityInitializationError):
                    session.update(**arguments)
            self.assertEqual(session.update(),1)
            self.assertEqual(len(self.records('ready')),1)

    def test_full_publish_and_explicit_replace_still_create_new_instances(self):
        cap = self.cap('read','same')
        with Session() as session:
            session.publish([cap])
            session.update(replace={cap.identity:cap})
            session.publish([cap])
        self.assertEqual(len(self.records('ready')),3)
        self.assertEqual(len({r['instance'] for r in self.records('ready')}),3)
        self.assertEqual(len(self.records('closed')),3)

    def test_multiple_updates_during_model_turn_keep_shared_resources_until_last_holder_ends(self):
        from nervus import Code, Finish, ScriptedModel
        kept, target = self.cap('cache','kept'), self.cap('read','old')
        changed = replace(target, factory=partial(SQLiteQuery,self.audit,'new'), implementation_version='new')
        with Session() as session:
            session.publish([kept,target])
            def update_from_host(context):
                session.update(replace={target.identity:changed})
                session.update(remove=[kept.identity])
                self.assertEqual(self.records('closed'),[])
                return Code("a = await tools.cache(2)\nb = await tools.read(2)", exports=('a','b'))
            result = session.run('use pinned resources', ScriptedModel([
                Code('a = await tools.cache(1)', exports=('a',)), update_from_host, Finish('done')]))
            self.assertEqual(result.feedback[-1].values['a']['count'],2)
            self.assertEqual(result.feedback[-1].values['b']['marker'],'old')
            self.assertCountEqual([r['marker'] for r in self.records('closed')],['old','kept'])
            session.begin_turn()
            self.assertEqual(session.execute('value = await tools.read(3)', exports=['value'])['value']['marker'],'new')
            session.end_turn()
        self.assertEqual(len(self.records('ready')),3)
        self.assertEqual(len(self.records('closed')),3)
