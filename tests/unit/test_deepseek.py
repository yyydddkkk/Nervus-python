from dataclasses import replace
import io
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from nervus import Code, Finish, ModelError, Session
from nervus.context import CapabilityView, Feedback, ModelContext
from nervus.models.deepseek import (
    DeepSeekFlash, DeepSeekResponseError, DeepSeekTimeoutError, DeepSeekTransportError,
)


USAGE = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
         "prompt_cache_hit_tokens": 40, "prompt_cache_miss_tokens": 60}


def response(content, finish_reason="stop", usage=USAGE):
    stream = io.BytesIO(json.dumps({"id": "request-1", "model": "deepseek-v4-flash",
                                  "usage": usage, "choices": [{"finish_reason": finish_reason,
                                      "message": {"role": "assistant", "content": content}}]}).encode())
    stream.status = 200
    return stream


class DeepSeekTests(unittest.TestCase):
    def context(self):
        return ModelContext("compute", 1, 2, 2, 5,
                            (CapabilityView("echo", "identity", "1", "v2", "(value)",
                                            "Return the provided value", "Same JSON value as input"),), (),
                            (Feedback(1, "answer = unknown", ("answer",), {}, "NameError: unknown"),))

    def test_request_rules_selected_feedback_action_and_usage(self):
        model = DeepSeekFlash("fake-key", timeout=11)
        with patch("nervus.models.deepseek.request.urlopen", return_value=response(
                '{"type":"code","code":"answer = 42","exports":["answer"]}')) as post:
            self.assertEqual(model.decide(self.context()), Code("answer = 42", ("answer",)))
        req = post.call_args.args[0]
        self.assertEqual(post.call_args.kwargs["timeout"], 11)
        self.assertEqual(req.full_url, "https://api.deepseek.com/chat/completions")
        body = json.loads(req.data)
        self.assertEqual(body["model"], "deepseek-v4-flash")
        self.assertEqual(body["thinking"], {"type": "disabled"})
        self.assertEqual(body["response_format"], {"type": "json_object"})
        rules = body["messages"][0]["content"]
        for rule in ("persist", "exports", "print()", "tools.", "conversation memory"):
            self.assertIn(rule, rules)
        context = json.loads(body["messages"][1]["content"])
        self.assertEqual(context["feedback"][0]["error"], "NameError: unknown")
        self.assertNotIn("namespace", context)
        self.assertEqual(context["capabilities"][0]["description"], "Return the provided value")
        self.assertEqual(context["capabilities"][0]["returns"], "Same JSON value as input")
        self.assertEqual(model.records[0].usage, USAGE)
        self.assertEqual(model.records[0].outcome, "succeeded")

    def test_finish_and_missing_usage_are_not_fabricated(self):
        model = DeepSeekFlash("fake-key")
        with patch("nervus.models.deepseek.request.urlopen", return_value=response(
                '{"type":"finish","answer":"42"}', usage=None)):
            self.assertEqual(model.decide(self.context()), Finish("42"))
        self.assertIsNone(model.records[0].usage)

    def test_invalid_actions_are_not_extracted_repaired_or_executed(self):
        for content in ('', '```json\n{"type":"finish","answer":"x"}\n```',
                        '{"type":"code","code":"bad = 1","exports":"bad"}',
                        '{"type":"finish","answer":42}',
                        '{"type":"finish","answer":"x","extra":true}',
                        '{"type":"finish","type":"code","answer":"x"}'):
            with self.subTest(content=content):
                model = DeepSeekFlash("fake-key")
                with patch("nervus.models.deepseek.request.urlopen", return_value=response(content)) as post:
                    with self.assertRaises(DeepSeekResponseError):
                        model.decide(self.context())
                self.assertEqual(post.call_count, 1)
                self.assertEqual(model.records[0].outcome, "response_error")
                self.assertEqual(model.records[0].usage, USAGE)

    def test_truncated_completion_retains_usage_but_is_not_an_action(self):
        model = DeepSeekFlash("fake-key")
        with patch("nervus.models.deepseek.request.urlopen", return_value=response(
                '{"type":"finish","answer":"truncated"}', finish_reason="length")):
            with self.assertRaises(DeepSeekResponseError):
                model.decide(self.context())
        self.assertEqual(model.records[0].finish_reason, "length")
        self.assertEqual(model.records[0].usage, USAGE)

    def test_transport_and_timeout_errors_do_not_retry_or_expose_response_body(self):
        cases = [(TimeoutError(), DeepSeekTimeoutError, "timeout"),
                 (URLError(TimeoutError()), DeepSeekTimeoutError, "timeout"),
                 (URLError("network down"), DeepSeekTransportError, "transport_error"),
                 (HTTPError("https://api.deepseek.com", 429, "rate limit", {}, io.BytesIO(b"fake-key")),
                  DeepSeekTransportError, "transport_error")]
        for error, expected, outcome in cases:
            with self.subTest(expected=expected, outcome=outcome):
                model = DeepSeekFlash("fake-key")
                with patch("nervus.models.deepseek.request.urlopen", side_effect=error) as post:
                    with self.assertRaises(expected) as caught:
                        model.decide(self.context())
                self.assertNotIn("fake-key", str(caught.exception))
                self.assertEqual(post.call_count, 1)
                self.assertEqual(model.records[0].outcome, outcome)
                self.assertIsNone(model.records[0].usage)

    def test_non_json_feedback_fails_before_network(self):
        context = replace(self.context(), feedback=(Feedback(1, "x = b'x'", ("x",), {"x": b"x"}),))
        model = DeepSeekFlash("fake-key")
        with patch("nervus.models.deepseek.request.urlopen") as post:
            with self.assertRaisesRegex(ModelError, "JSON-serializable"):
                model.decide(context)
        post.assert_not_called()
        self.assertEqual(model.records[0].outcome, "input_error")

    def test_adapter_does_not_add_prior_turn_conversation(self):
        model = DeepSeekFlash("fake-key")
        with patch("nervus.models.deepseek.request.urlopen", side_effect=[
            response('{"type":"finish","answer":"one"}'),
            response('{"type":"finish","answer":"two"}')]) as post:
            model.decide(self.context())
            model.decide(replace(self.context(), turn=2, feedback=(), input="use named variable saved"))
        second = json.loads(post.call_args.args[0].data)
        self.assertEqual(len(second["messages"]), 2)
        self.assertNotIn("answer = unknown", second["messages"][1]["content"])

    def test_parse_failure_closes_formal_turn_without_executing_text(self):
        with Session() as session:
            with patch("nervus.models.deepseek.request.urlopen", return_value=response(
                    '{"type":"code","code":"bad = 1","exports":"bad"}')):
                with self.assertRaises(DeepSeekResponseError):
                    session.run("invalid response", DeepSeekFlash("fake-key"))
            session.begin_turn()
            session.execute("assert 'bad' not in globals()")
            session.end_turn()
