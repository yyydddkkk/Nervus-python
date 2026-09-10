"""One non-streaming DeepSeek V4 Flash Adapter, with no automatic retries."""

from dataclasses import asdict, dataclass
import json
import http.client
import math
import os
import time
from urllib import error, request
from urllib.parse import urlsplit

from .model import Code, Finish
from ..context import ModelContext
from ..errors import ModelError


EXECUTION_RULES = """You are operating Nervus, a code-driven Python agent kernel.
Respond with exactly one JSON object, never Markdown or a code fence:
{"type":"code","code":"answer = 6 * 7","exports":["answer"]}
or {"type":"finish","answer":"Your final answer"}.

Execution rules:
- Variables, ordinary functions and results persist across code calls AND Turns
  in this Session. Code runs at module scope; top-level await is supported.
- Running tasks must finish within their Turn. Use await to observe their results.
- Invoke capabilities only via await tools.<name>(...), using the supplied Turn
  capability signatures, purpose and return-structure descriptions. Saved aliases
  follow this Turn's identity/version binding.
- Values selected in exports are sent back as structured data. Choose exports yourself;
  the Host does not know your variable names. Exports names must be strings.
- Bounded stdout/stderr (including print()) is provided in context.output, even if
  code fails later. Chunks are labeled by originating execution, NOT the entry in
  which they happened to be collected. Background tasks retain their creation origin.
  Check the truncation flag. Bare expression values are not automatically displayed.
  Use prints to inspect unknown structures; use exports for structured results.
  Do not export functions, modules, Tasks or live handles.
- An empty exports list returns no structured data. Inspect execution feedback before claiming
  results. On code error, use the next decision to correct it; do not suppress errors
  or assume earlier variable modifications were rolled back.
- Return finish when done. Do not put code in the final answer as a substitute for
  executing it. Do not invent capability results.
- The Kernel does not automatically include prior Turn conversations or the whole
  namespace. A Host may explicitly supply conversation history inside Input; use
  that provided history, but never invent unprovided dialogue. Persistent Python
  state does NOT itself mean conversation memory. Reuse saved variables/functions
  when identified or discovered. Capability changes apply to the supplied snapshot.
- To discover existing names when needed, execute directory = inspect_workspace()
  and export or print directory. Optional max_entries, max_bytes and prefix bound
  or filter this metadata-only view. It does not return variable contents. Inspect
  contents yourself in code when useful. A saved capability reference can exist
  while unavailable in this Turn; check its availability. No directory is sent
  automatically on each decision.
"""


class DeepSeekTimeoutError(ModelError):
    pass


class DeepSeekTransportError(ModelError):
    pass


class DeepSeekResponseError(ModelError):
    pass


@dataclass(frozen=True)
class DeepSeekCall:
    number: int
    turn: int
    revision: int
    step: int
    outcome: str
    elapsed_seconds: float
    http_status: int | None
    response_id: str | None
    response_model: str | None
    finish_reason: str | None
    usage: dict | None


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON object field")
        value[key] = item
    return value


def _decode(text, *, strict=True):
    def reject_constant(value):
        raise ValueError("Non-finite JSON number")
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=reject_constant, strict=strict)


def _decode_action(content):
    try:
        return _decode(content)
    except json.JSONDecodeError as exc:
        # Some responses put literal LF/CR/TAB inside JSON strings. Preserve
        # those characters exactly; do not repair quotes, fields or code fences.
        if (not exc.msg.startswith("Invalid control character")
                or content[exc.pos:exc.pos + 1] not in ("\n", "\r", "\t")
                or any(ord(char) < 32 and char not in "\n\r\t" for char in content)):
            raise
        return _decode(content, strict=False)


def _action(content):
    def reject(reason):
        error = DeepSeekResponseError(f"Invalid Code/Finish JSON action: {reason}")
        error.diagnostic = {"reason": reason}
        raise error

    try:
        value = _decode_action(content)
    except json.JSONDecodeError as exc:
        reject(f"invalid JSON at line {exc.lineno}, column {exc.colno}")
    except (ValueError, TypeError) as exc:
        reject(str(exc))
    if not isinstance(value, dict):
        reject("action must be an object")
    kind = value.get("type")
    if kind not in ("code", "finish"):
        reject("type must be code or finish")
    required = {"type", "code", "exports"} if kind == "code" else {"type", "answer"}
    missing, extra = required - value.keys(), value.keys() - required
    if missing:
        reject("missing fields: " + ", ".join(sorted(missing)))
    if extra:
        reject("unexpected fields: " + ", ".join(sorted(extra))[:256])
    if kind == "code":
        if not isinstance(value["code"], str):
            reject("code must be a string")
        if (not isinstance(value["exports"], list)
                or not all(isinstance(name, str) for name in value["exports"])):
            reject("exports must be a list of strings")
        return Code(value["code"], tuple(value["exports"]))
    if not isinstance(value["answer"], str):
        reject("answer must be a string")
    return Finish(value["answer"])


class DeepSeekFlash:
    model = "deepseek-v4-flash"

    def __init__(self, api_key: str | None = None, *, base_url="https://api.deepseek.com",
                 timeout: float = 30, max_tokens: int = 1536, debug_responses: bool = False, trace=None):
        self._api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self._api_key or not self._api_key.strip():
            raise ValueError("DEEPSEEK_API_KEY is required")
        url = urlsplit(base_url)
        if (url.scheme != "https" or not url.hostname or url.username or url.password
                or url.query or url.fragment):
            raise ValueError("base_url must be an HTTPS API base without credentials or query")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        self._endpoint = base_url.rstrip("/") + "/chat/completions"
        self.trace = trace
        self.trace_errors = []
        self.debug_responses = debug_responses
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.records: list[DeepSeekCall] = []

    def _trace(self, event):
        if self.trace is None:
            return
        try:
            # Only selected payload fields, never headers or credentials.
            def redact(value):
                if isinstance(value, str):
                    return value.replace(self._api_key, "<REDACTED>")
                if isinstance(value, list):
                    return [redact(item) for item in value]
                if isinstance(value, dict):
                    return {redact(key): redact(item) for key, item in value.items()}
                return value
            self.trace(redact(event))
        except Exception as error:
            self.trace_errors.append(type(error).__name__)

    def decide(self, context: ModelContext) -> Code | Finish:
        started = time.monotonic()
        outcome = "input_error"
        status = response_id = response_model = finish_reason = usage = None
        try:
            try:
                content = json.dumps(asdict(context), ensure_ascii=False, allow_nan=False)
                payload = json.dumps({
                    "model": self.model, "stream": False,
                    "thinking": {"type": "disabled"}, "temperature": 0,
                    "response_format": {"type": "json_object"}, "max_tokens": self.max_tokens,
                    "messages": [{"role": "system", "content": EXECUTION_RULES},
                                 {"role": "user", "content": content}],
                }, ensure_ascii=False).encode("utf-8")
            except (TypeError, ValueError) as exc:
                raise ModelError("DeepSeek context must contain JSON-serializable finite data") from exc
            req = request.Request(self._endpoint, data=payload, method="POST", headers={
                "Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json",
            })
            outcome = "transport_error"
            try:
                # urllib timeout bounds blocking connect/read operations, NOT the
                # entire wall-clock duration of a trickling response or DNS lookup.
                self._trace({"kind": "request", "body": json.loads(req.data),
                             "endpoint": self._endpoint, "timeout": self.timeout})
                with request.urlopen(req, timeout=self.timeout) as response:
                    status = response.status
                    raw = response.read(2_000_001)
            except error.HTTPError as exc:
                status = exc.code
                exc.close()
                raise DeepSeekTransportError(f"DeepSeek HTTP {status}; request not retried") from None
            except (TimeoutError, error.URLError, OSError, http.client.HTTPException) as exc:
                if isinstance(exc, TimeoutError) or isinstance(getattr(exc, "reason", None), TimeoutError):
                    outcome = "timeout"
                    raise DeepSeekTimeoutError("DeepSeek network operation timed out; not retried") from None
                raise DeepSeekTransportError("DeepSeek network request failed; not retried") from None
            outcome = "response_error"
            try:
                if len(raw) > 2_000_000:
                    raise ValueError("Response exceeds the Adapter limit")
                document = _decode(raw)
                self._trace({"kind": "response", "body": document})
                if not isinstance(document, dict):
                    raise ValueError("Response must be an object")
                response_id = document.get("id")
                response_model = document.get("model")
                supplied_usage = document.get("usage")
                if supplied_usage is not None:
                    if not isinstance(supplied_usage, dict):
                        raise ValueError("Invalid usage object")
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                        if type(supplied_usage.get(key)) is not int or supplied_usage[key] < 0:
                            raise ValueError("Invalid token usage")
                    usage = supplied_usage  # Retain cache/reasoning details when present.
                choices = document["choices"]
                if not isinstance(choices, list) or len(choices) != 1:
                    raise ValueError("Expected one choice")
                choice = choices[0]
                finish_reason = choice["finish_reason"]
                if finish_reason != "stop":
                    raise ValueError("Incomplete/unsupported completion reason")
                message = choice["message"]
                if (message.get("role") != "assistant" or message.get("tool_calls")
                        or not isinstance(message.get("content"), str) or not message["content"].strip()):
                    raise ValueError("Missing text action or unexpected tool calls")
                try:
                    action = _action(message["content"])
                except DeepSeekResponseError as exc:
                    # Capture the rejected model content only on explicit debug;
                    # never capture request headers or transport error bodies.
                    reason = exc.diagnostic["reason"].replace(self._api_key, "<REDACTED>")
                    exc.args = (f"Invalid Code/Finish JSON action: {reason}",)
                    exc.diagnostic = {"reason": reason, "turn": context.turn, "step": context.step}
                    if self.debug_responses:
                        content = message["content"].replace(self._api_key, "<REDACTED>")
                        exc.diagnostic.update(content=content[:8192], content_truncated=len(content) > 8192)
                    raise
            except (ValueError, TypeError, KeyError, UnicodeError, AttributeError) as exc:
                raise DeepSeekResponseError("Malformed or incomplete DeepSeek response") from exc
            outcome = "succeeded"
            return action
        finally:
            self.records.append(DeepSeekCall(
                len(self.records) + 1, context.turn, context.revision, context.step, outcome,
                round(time.monotonic() - started, 6), status, response_id, response_model,
                finish_reason, usage,
            ))
