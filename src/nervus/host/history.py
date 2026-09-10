"""Explicit conversation retention and request budgets; no silent pruning."""

from copy import deepcopy
from dataclasses import asdict, replace
import json

from ..errors import ModelError


def encoded_size(value):
    return len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8'))


class ContextBudgetError(ModelError):
    pass


class History:
    def __init__(self, limit):
        self.limit = limit
        self.messages = []
        self.cleared = False

    @property
    def size(self):
        return encoded_size(self.messages)

    def begin(self, text):
        message = {'role': 'user', 'content': text}
        if encoded_size([*self.messages, message]) > self.limit:
            raise ContextBudgetError('聊天历史预算不足；请 /clear-history 或 /new。消息尚未提交。')
        prior = deepcopy(self.messages)
        self.messages.append(message)
        return prior

    def finish(self, text):
        self.messages.append({'role': 'assistant', 'content': text})
        return self.size > self.limit

    def clear(self):
        self.messages.clear()
        self.cleared = True


def model_context(context, history, cwd, limit, cleared=False):
    envelope = {
        'working_directory': str(cwd),
        'host_rules': (
            'You are an interactive terminal agent. Respond to the user in their language. '
            'The conversation below is explicitly provided history, separate from persistent Python state. '
            'Use Python to inspect and modify files in working_directory; pathlib/open are available. '
            'Keep the worker cwd unchanged. Run ALL external programs via await tools.run(argv); '
            'do not use subprocess, os.system, exec/spawn or alternative process APIs in Python. '
            'tools.run uses the Host working directory; use explicit argv and check returncode and '
            'stdout_truncated/stderr_truncated. Never assume a program succeeded. '
            'Use inspect_workspace() when you need to discover saved objects. '
            'Only final user-facing answers and status notices persist in chat history; execution '
            'details are in this Turn feedback. Finish with a useful, concise answer describing results. '
            'Do not ask for routine approval; act within the user task. No autonomous capability registration.'
        ),
        'history_cleared_by_user': cleared,
        'conversation': history,
        'task': context.input,
    }
    enriched = replace(context, input=json.dumps(envelope, ensure_ascii=False))
    size = encoded_size(asdict(enriched))
    if size > limit:
        raise ContextBudgetError(f'本轮上下文 {size} 字节超过 {limit} 上限；未发送这次模型请求。请减少导出/输出或清理聊天历史。')
    return enriched
