"""A fixed, manually driven 4→3→4 experiment. No model-facing instructions."""
from dataclasses import asdict
import json
from pathlib import Path
import threading

from .. import Capability, Finish

QUESTION = '你当前可通过 tools 调用哪些能力？请列出全部名称，并给出总数。'


class Echo:
    async def initialize(self): pass
    async def close(self): pass
    async def __call__(self, value): return value


class Add:
    async def initialize(self): pass
    async def close(self): pass
    async def __call__(self, a, b): return a + b


class Upper:
    async def initialize(self): pass
    async def close(self): pass
    async def __call__(self, text): return text.upper()


def metadata(capability):
    return {key: getattr(capability, key) for key in
            ('name', 'identity', 'interface_version', 'implementation_version', 'description', 'returns')}


class Experiment:
    def __init__(self, path, session, run_capability, configuration, secrets=()):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open('x', encoding='utf-8')  # Never overwrite an earlier experiment.
        self.lock = threading.RLock()
        self.errors = []
        self.secrets = tuple(s for s in secrets if s)
        self.session = session
        self.capabilities = (
            run_capability,
            Capability('echo', Echo, description='Return a supplied value.', returns='The supplied plain-data value.'),
            Capability('add', Add, description='Add two values.', returns='The sum of a and b.'),
            Capability('upper', Upper, description='Convert text to uppercase.', returns='Uppercase string.'),
        )
        self.published = None
        self.trials = []
        self.write({'event': 'configuration', 'session': session.identity, 'question': QUESTION,
                    'sequence': [4, 3, 4], 'configuration': configuration,
                    'fixed_capabilities': [metadata(c) for c in self.capabilities]})

    def redact(self, value):
        if isinstance(value, str):
            for secret in self.secrets:
                value = value.replace(secret, '<REDACTED>')
            return value
        if isinstance(value, dict):
            return {self.redact(k): self.redact(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.redact(v) for v in value]
        return value

    def write(self, event):
        # A measurement failure must not replace the model's answer.
        try:
            with self.lock:
                self.stream.write(json.dumps(self.redact({'session': self.session.identity, **event}), ensure_ascii=False, default=str) + '\n')
                self.stream.flush()
        except Exception as error:
            self.errors.append(type(error).__name__)

    def publish(self, count):
        if count not in (3, 4):
            raise ValueError('Only the fixed 3- or 4-capability environments are supported')
        capabilities = self.capabilities if count == 4 else self.capabilities[:3]
        revision = self.session.publish(capabilities)
        self.published = {'revision': revision, 'capabilities': [metadata(c) for c in capabilities]}
        self.write({'event': 'publication', 'session': self.session.identity, **self.published})
        return [c.name for c in capabilities]

    def ready(self):
        if len(self.trials) >= 3:
            raise ValueError('三轮已完成；不自动重跑。请 /cap-report 核对或 /quit。')
        required = (4, 3, 4)[len(self.trials)]
        if self.published is None or len(self.published['capabilities']) != required:
            raise ValueError(f'请先执行 /cap-env {required}，然后 /cap-ask。')
        if self.errors:
            raise ValueError('实验日志写入失败；不再提交新的实验请求。')

    def begin(self):
        self.ready()
        trial = Trial(self, len(self.trials) + 1, self.published)
        self.trials.append(trial)
        return trial

    def check(self, count, names, looked='unknown'):
        if not self.trials or self.trials[-1].status != 'finished':
            raise ValueError('没有可核对的已完成回答；失败/停止不能当作数量回答。')
        trial = self.trials[-1]
        expected = [cap['name'] for cap in trial.publication['capabilities']]
        trial.check = {'source': 'host_interpretation_of_raw_answer', 'declared_count': count,
                       'declared_names': names, 'expected_count': len(expected), 'expected_names': expected,
                       'names_match': sorted(names) == sorted(expected), 'count_match': count == len(expected),
                       'claim_internally_consistent': count == len(names) == len(set(names)),
                       'other_code_inspection_review': looked}
        self.write({'event': 'answer_check', 'trial': trial.number, **trial.check})
        return trial.check

    def report(self):
        return {'session': self.session.identity, 'recording_errors': list(self.errors),
                'actual_counts': [len(t.publication['capabilities']) for t in self.trials],
                'host_interpreted_answer_counts': [t.check['declared_count'] if t.check else None for t in self.trials],
                'trials': [{'trial': t.number, 'turn': t.turn, 'status': t.status,
                            'actual_names': [c['name'] for c in t.publication['capabilities']],
                            'answer': t.answer, 'inspection': t.inspection, 'check': t.check}
                           for t in self.trials]}


class Trial:
    def __init__(self, experiment, number, publication):
        self.experiment, self.number, self.publication = experiment, number, publication
        self.turn = None
        self.status = 'running'
        self.answer = None
        self.check = None
        self.code_steps = 0
        self.inspection = None
        self.write('trial_started', publication=publication, input=QUESTION)

    def write(self, event, **values):
        self.experiment.write({'event': event, 'trial': self.number, 'turn': self.turn,
                               'late': self.status != 'running' and event != 'trial_finished', **values})

    def wire(self, event):
        self.write('adapter_trace', trace=event)

    def context(self, context):
        self.turn = context.turn
        self.write('model_context', context=asdict(context),
                   history_in_input=json.loads(context.input)['conversation'])

    def action(self, action, cancelled):
        if not isinstance(action, Finish):
            self.code_steps += 1
        self.write('model_action', action_type=type(action).__name__, action=asdict(action),
                   cancelled_at_return=cancelled)

    def finish(self, status, result=None, error=None, model=None):
        if self.status != 'running':
            return
        self.status = status
        if result is not None:
            self.turn = result.turn
            self.answer = result.answer
        try:
            observed = [e for e in self.experiment.session.inspections if e['turn'] == self.turn]
            self.inspection = {'workspace_api_calls': observed,
                               'code_inspection': 'yes' if observed else ('no_code' if not self.code_steps else 'needs_review'),
                               'note': 'Other mechanisms, including dir(), require review of executed code and feedback.'}
        except Exception as exc:
            self.inspection = {'code_inspection': 'unknown', 'error': type(exc).__name__}
        self.write('trial_finished', status=status, result=asdict(result) if result else None,
                   error=str(error) if error else None, raw_answer=self.answer, inspection=self.inspection,
                   calls=[c for c in self.experiment.session.calls if c['turn'] == self.turn],
                   model_records=[asdict(r) for r in getattr(model, 'records', ())],
                   adapter_trace_errors=list(getattr(model, 'trace_errors', ())))
