"""Offline evidence for rejected actions; not a replay of the user's missing response."""
import unittest
from unittest.mock import patch

from nervus import ModelContext
from nervus.models.deepseek import DeepSeekFlash, DeepSeekResponseError
from tests.unit.test_deepseek import response


class DiagnosticsTests(unittest.TestCase):
    def test_schema_failure_explains_reason_without_accepting_or_retrying(self):
        model = DeepSeekFlash('offline-key')
        context = ModelContext('你能做些什么', 2, 1, 1, 20, (), (), ())
        with patch('nervus.models.deepseek.request.urlopen', return_value=response(
                '{"type":"finish","answer":"help","extra":true}')) as request:
            with self.assertRaises(DeepSeekResponseError) as caught:
                model.decide(context)
        self.assertIn('unexpected fields', str(caught.exception))
        self.assertEqual(caught.exception.diagnostic['reason'], 'unexpected fields: extra')
        self.assertNotIn('content', caught.exception.diagnostic)
        self.assertEqual(request.call_count, 1)

    def test_debug_retains_bounded_rejected_content_without_api_key(self):
        model = DeepSeekFlash('offline-key', debug_responses=True)
        context = ModelContext('task', 1, 1, 1, 20, (), (), ())
        content = '{"type":"finish","answer":42,"note":"offline-key' + 'x'*9000 + '"}'
        with patch('nervus.models.deepseek.request.urlopen', return_value=response(content)):
            with self.assertRaises(DeepSeekResponseError) as caught:
                model.decide(context)
        diagnostic = caught.exception.diagnostic
        self.assertTrue(diagnostic['content_truncated'])
        self.assertLessEqual(len(diagnostic['content']), 8192)
        self.assertNotIn('offline-key', str(diagnostic))

    def test_json_syntax_and_field_types_are_distinguished(self):
        for content, expected in [('not json', 'invalid JSON'),
                                  ('{"type":"finish","answer":42}', 'answer must be a string'),
                                  ('{"type":"code","code":"x=1","exports":"x"}', 'exports must be a list of strings')]:
            with self.subTest(content=content):
                model = DeepSeekFlash('offline-key')
                with patch('nervus.models.deepseek.request.urlopen', return_value=response(content)):
                    with self.assertRaises(DeepSeekResponseError) as caught:
                        model.decide(ModelContext('task', 1, 1, 1, 20, (), (), ()))
                self.assertIn(expected, str(caught.exception))
