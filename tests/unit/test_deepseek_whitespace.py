"""Synthetic malformed JSON; only raw LF/CR/TAB get lexical compatibility."""
from pathlib import Path
import unittest
from unittest.mock import patch

from nervus import Code, Finish, ModelContext, Session
from nervus.models.deepseek import DeepSeekFlash, DeepSeekResponseError, _action
from tests.unit.test_deepseek import response


CONTENT = (Path(__file__).parents[1] / 'fixtures/deepseek_unescaped_answer.txt').read_text()


class WhitespaceTests(unittest.TestCase):
    def test_synthetic_response_finishes_without_code_execution_or_retry(self):
        expected = CONTENT[len('{"type":"finish","answer":"'):-3]
        with Session() as session:
            model = DeepSeekFlash('offline-key')
            with patch('nervus.models.deepseek.request.urlopen', return_value=response(CONTENT)) as post:
                result = session.run('Describe the synthetic example tasks', model)
            self.assertEqual(result.reason, 'finished')
            self.assertEqual(result.answer, expected)
            self.assertEqual(result.feedback, ())
            self.assertEqual(session.calls, ())
            self.assertEqual(post.call_count, 1)
            self.assertEqual(model.records[0].outcome, 'succeeded')

    def test_raw_whitespace_preserves_answer_and_code_exactly(self):
        self.assertEqual(_action('{"type":"finish","answer":"a\nb\r\nc\td"}'), Finish('a\nb\r\nc\td'))
        self.assertEqual(_action('{"type":"code","code":"a = 1\nb = 2","exports":[]}'), Code('a = 1\nb = 2'))
        self.assertEqual(_action(r'{"type":"finish","answer":"literal \\n, escaped \n"}'), Finish('literal \\n, escaped \n'))

    def test_compatibility_does_not_guess_structure_or_accept_other_controls(self):
        for content in (
            '{"type":"finish","answer":"a\nb","extra":true}',
            '{"type":"finish","answer":"a\nb","type":"finish"}',
            '{"type":"finish","answer":"a\nb"',
            '{"type":"finish","answer":"a\nb"} trailing',
            '{"type":"finish","answer":"a\nb"broken"}',
            '{"type":"finish","answer":"a\nb\x00"}',
            '{"type":"finish","answer":"a\nb\x1b"}',
            '{"type":"code","code":"a = 1\nb = 2","exports":"a"}',
            '```json\n{"type":"finish","answer":"a\nb"}\n```',
        ):
            with self.subTest(content=content), self.assertRaises(DeepSeekResponseError):
                _action(content)

    def test_http_envelope_remains_strict(self):
        import io
        raw = b'{"id":"bad\nvalue","choices":[]}'
        stream = io.BytesIO(raw)
        stream.status = 200
        with patch('nervus.models.deepseek.request.urlopen', return_value=stream):
            with self.assertRaises(DeepSeekResponseError):
                DeepSeekFlash('offline-key').decide(ModelContext('task', 1, 1, 1, 20, (), (), ()))
