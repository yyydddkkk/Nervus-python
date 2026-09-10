"""A terminal Host using the existing Kernel; no model calls on import."""

import argparse
from contextlib import chdir
from dataclasses import asdict
from functools import partial
import json
import math
import os
from pathlib import Path
import queue
import re
import signal
import threading
import time

from .. import Capability, Code, Finish, ProcessRunner, ScriptedModel, Session
from ..errors import ModelError, NervusError, ProcessCleanupError, SessionStateError, WorkingStateLostError
from ..models.deepseek import DeepSeekFlash
from .history import ContextBudgetError, History, model_context
from .capability_experiment import Experiment, QUESTION


def load_env(path):
    """Literal dotenv subset; explicit path only, existing environment wins."""
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('export '):
            line = line[7:].strip()
        key, sep, value = line.partition('=')
        key, value = key.strip(), value.strip()
        if not sep or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key):
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def read_input(events):
    try:
        import readline  # Standard-library line editing when available.
    except ImportError:
        pass
    while True:
        try:
            line = input()
        except EOFError:
            events.put((None, 'input', '/quit'))
            return
        events.put((None, 'input', line))
        if line.strip() == '/quit':
            return


def show_debug(label, value):
    print(f'[debug:{label}] {json.dumps(value, ensure_ascii=False, default=str)}', flush=True)


def show_python(kind, value):
    if kind == 'code':
        print(f"[Python · Turn {value['turn']} · 第 {value['step']} 步 · 提交执行]", flush=True)
        print('```python\n' + value['code'] + '\n```', flush=True)
        print('导出：' + (', '.join(value['exports']) or '无'), flush=True)
    elif kind == 'feedback':
        print(f"[Python · 执行 {value['execution']} · {'失败' if value['error'] else '完成'}]", flush=True)
        if value['error']:
            print(value['error'], flush=True)
        if value['values']:
            text = json.dumps(value['values'], ensure_ascii=False, default=str, indent=2)
            print(text[:2048] + ('\n[显示已截断；不影响模型收到的反馈]' if len(text) > 2048 else ''), flush=True)
    elif kind == 'output':
        for chunk in value['chunks']:
            print(f"[{chunk['channel']} · 执行 {chunk['execution']}]", flush=True)
            print(chunk['text'], end='' if chunk['text'].endswith('\n') else '\n', flush=True)
        if value.get('truncated'):
            print('[Python 输出已达到内核上限，部分内容未保留]', flush=True)


class Job:
    def __init__(self, session, model, text, prior, history_cleared, cwd, events, args):
        self.session, self.model, self.text = session, model, text
        self.prior, self.history_cleared, self.cwd = prior, history_cleared, cwd
        self.events, self.args = events, args
        self.show_python = getattr(args, "show_python", False) and not args.debug
        self.cancelled = False
        self.error = None
        self.turn = None
        self.trial = None
        self.feedback_seen = 0
        self.output_lengths = []
        self.output_dropped = 0
        self.thread = threading.Thread(target=self.run, daemon=True)

    def emit(self, kind, value):
        self.events.put((self, kind, value))

    def observe(self, feedback, output):
        for entry in feedback[self.feedback_seen:]:
            if entry.error and not self.show_python:
                self.emit('notice', '代码反馈：' + entry.error.splitlines()[0])
            if self.args.debug:
                self.emit('debug', ('feedback', asdict(entry)))
            elif self.show_python:
                self.emit('python', ('feedback', asdict(entry)))
        self.feedback_seen = len(feedback)
        if (self.args.debug or self.show_python) and output:
            chunks = output.get('chunks', [])
            delta = []
            for index, chunk in enumerate(chunks):
                previous = self.output_lengths[index] if index < len(self.output_lengths) else 0
                if len(chunk['text']) > previous:
                    delta.append({**chunk, 'text': chunk['text'][previous:]})
            dropped = output.get('dropped_characters', 0)
            if delta or dropped != self.output_dropped:
                self.emit('debug' if self.args.debug else 'python', ('output', {**output, 'chunks': delta}))
            self.output_lengths = [len(chunk['text']) for chunk in chunks]
            self.output_dropped = dropped

    def decide(self, context):
        self.turn = context.turn
        if self.cancelled:
            raise ModelError('Host requested stop before the next model decision')
        self.observe(context.feedback, context.output)
        enriched = model_context(context, self.prior, self.cwd, self.args.context_bytes, self.history_cleared)
        if self.trial is not None:
            self.trial.context(enriched)
        if self.args.debug and context.step == 1:
            self.emit('debug', ('input', json.loads(enriched.input)))
        self.emit('notice', f'思考 · 第 {context.step} 步')
        action = self.model.decide(enriched)
        if self.trial is not None:
            self.trial.action(action, self.cancelled)
        if not self.cancelled:
            if self.args.debug:
                self.emit('debug', ('action', asdict(action)))
            if isinstance(action, Code):
                if self.show_python:
                    self.emit('python', ('code', {'turn': context.turn, 'step': context.step,
                                                 'code': action.code, 'exports': action.exports}))
                if not self.show_python:
                    self.emit('notice', f'执行 Python · 第 {context.step} 步（/stop 可停止）')
        return action

    def run(self):
        try:
            result = self.session.run(self.text, self, max_decisions=self.args.max_decisions,
                                      call_budget=self.args.call_budget)
            self.observe(result.feedback, result.output)
            self.emit('result', result)
        except Exception as error:
            self.error = error
            if self.args.debug and hasattr(error, 'diagnostic'):
                self.emit('debug', ('rejected-response', error.diagnostic))
            self.emit('error', error)

    def stop(self):
        self.cancelled = True
        if isinstance(self.error, (WorkingStateLostError, ProcessCleanupError)):
            raise self.error
        turn = self.turn if self.turn is not None else self.session.active_turn
        if turn is None:
            return None
        try:
            return self.session.stop(turn, grace=self.args.stop_grace)
        except SessionStateError:
            if self.session.active_turn is None and not self.thread.is_alive():
                return None
            raise


def offline_factory(path):
    document = json.loads(path.read_text())
    turns = []
    for turn in document['turns']:
        actions = []
        for item in turn:
            if item.get('type') == 'code' and set(item) == {'type', 'code', 'exports'}:
                actions.append(Code(item['code'], tuple(item['exports'])))
            elif item.get('type') == 'finish' and set(item) == {'type', 'answer'}:
                actions.append(Finish(item['answer']))
            else:
                raise ValueError('Invalid offline Code/Finish action')
        turns.append(actions)
    iterator = iter(turns)
    def create():
        try:
            return ScriptedModel(next(iterator))
        except StopIteration:
            raise ModelError('离线脚本已用完；不会切换到真实模型。') from None
    return create


def process_capability():
    return Capability('run', partial(ProcessRunner, output_limit=8192),
            description='Run an argv list in the fixed Host working directory, without Shell parsing. '
                        'Await exit; use this capability for every external program.',
            returns='dict: returncode (negative for signal), stdout, stderr, '
                    'stdout_truncated, stderr_truncated. Each stream retains at most 8192 bytes.')


def create_session(args):
    # POSIX children inherit SIG_IGN across exec. Terminal Ctrl+C must reach the
    # Host stop path, not independently interrupt the worker's Python runtime.
    handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        session = Session(execution_timeout=args.execution_timeout)
    finally:
        signal.signal(signal.SIGINT, handler)
    try:
        session.publish([process_capability()])
        return session
    except Exception:
        session.close()
        raise


def error_text(error, debug=False):
    if isinstance(error, NervusError) or debug:
        return f'{type(error).__name__}: {error}'
    return f'{type(error).__name__}（使用 --debug 查看详情）'


def interact(args, factory, events):
    history = History(args.history_bytes)
    session = job = None
    experiment = None
    usable = True
    pending = None
    interrupted = False
    exit_code = 0

    def on_interrupt(signum, frame):
        nonlocal interrupted
        interrupted = True  # No locks or cleanup in the signal handler.

    previous_handler = signal.signal(signal.SIGINT, on_interrupt)

    def prompt():
        print('你> ', end='', flush=True)

    def finish_history(text):
        if history.finish(text):
            print('历史已超预算，完整回答已保留；下一任务需先 /clear-history 或 /new。', flush=True)

    def finish_trial(completed, status, result=None, error=None):
        if completed.trial is not None:
            completed.trial.finish(status, result, error, completed.model)
            if experiment.errors or getattr(completed.model, 'trace_errors', []):
                print('审计记录有错误，本轮不能据此得出完整结论。', flush=True)
            else:
                print('实验记录已保存；/cap-check <回答总数> <回答列出的名称...> 分别核对名单和数量。', flush=True)

    def debug_records(completed):
        if args.debug:
            show_debug('calls', [c for c in completed.session.calls if c['turn'] == completed.turn])
            show_debug('processes', [e for e in completed.session.process_events if e.get('turn') == completed.turn])
            show_debug('usage', [asdict(r) for r in getattr(completed.model, 'records', ())])

    try:
        session = create_session(args)
        if getattr(args, 'cap_log', None):
            from hashlib import sha256
            import sys
            from ..models.deepseek import EXECUTION_RULES
            configuration = {name: getattr(args, name) for name in
                ('history_bytes', 'context_bytes', 'max_decisions', 'call_budget', 'max_tokens',
                 'execution_timeout', 'stop_grace', 'debug', 'show_python', 'cwd')}
            configuration['python_version'] = sys.version
            configuration['source_sha256_at_start'] = {str(path): sha256(path.read_bytes()).hexdigest() for path in
                (Path(__file__), Path(__file__).with_name('history.py'), Path(__file__).parents[1] / 'models/deepseek.py')}
            configuration.update(model='ScriptedModel (offline)' if args.offline_script else DeepSeekFlash.model,
                                 thinking={'type': 'disabled'}, temperature=0, timeout=30,
                                 response_format={'type': 'json_object'}, tools_run_output_bytes=8192,
                                 execution_rules_sha256=sha256(EXECUTION_RULES.encode()).hexdigest(),
                                 history_strategy='retain user messages, final answers and status notices; explicit byte caps')
            experiment = Experiment(args.cap_log, session, process_capability(), configuration,
                                    secrets=(os.environ.get('DEEPSEEK_API_KEY', ''),))
        print(f'Nervus · {args.cwd}\n/help 查看命令 · Ctrl+C 停止 · /quit 退出', flush=True)
        if args.offline_script:
            print('离线脚本模式：动作预设，不调用真实模型。', flush=True)
        if experiment is not None:
            print('固定能力实验：/cap-env 4 → /cap-ask → /cap-env 3 → /cap-ask → /cap-env 4 → /cap-ask。控制提示不进入模型。', flush=True)
        prompt()
        threading.Thread(target=read_input, args=(events,), daemon=True).start()
        while True:
            if interrupted:
                interrupted = False
                if job:
                    if not job.cancelled:
                        print('正在停止当前 Turn…', flush=True)
                    job.cancelled = True
                else:
                    print('没有活动任务。', flush=True)
                    prompt()
            if job is not None and job.cancelled:
                try:
                    report = job.stop()
                    if isinstance(job.error, (WorkingStateLostError, ProcessCleanupError)):
                        raise job.error
                    if report is not None or not job.thread.is_alive():
                        notice = '任务已由用户停止；已完成的变量修改和文件副作用不会回滚。'
                        finish_history(notice)
                        print('已收口。Python 状态保留；迟到模型响应不再生效。', flush=True)
                        if report and report.get('release_errors'):
                            print('资源释放错误：' + '; '.join(report['release_errors']), flush=True)
                        finish_trial(job, 'stopped')
                        debug_records(job)
                        job = None
                        if pending is None:
                            prompt()
                except Exception as error:
                    usable = False
                    finish_history('停止未正常完成：' + error_text(error) + '。不要假设工作状态仍存在或重试未知调用。')
                    print('停止错误：' + error_text(error, args.debug) + '\n当前 Session 不可继续，请 /new 或 /quit。', flush=True)
                    finish_trial(job, 'error', error=error)
                    debug_records(job)
                    job = None
                    if pending is None:
                        prompt()
            if pending and job is None:
                if pending == 'quit':
                    break
                try:
                    if session is not None:
                        session.close()
                    session = None
                    history = History(args.history_bytes)
                    session = create_session(args)
                    usable = True
                    print('新 Session 已创建：聊天历史、变量和函数已清空；文件保留。', flush=True)
                except Exception as error:
                    usable = False
                    print('新建失败：' + error_text(error, args.debug), flush=True)
                pending = None
                prompt()
            try:
                sender, kind, value = events.get(timeout=0.05)
            except queue.Empty:
                continue
            if sender is not None:
                if sender is not job or sender.cancelled:
                    continue  # An old job cannot append history, display answers, or touch a new Session.
                if kind == 'notice':
                    print(value, flush=True)
                elif kind == 'debug':
                    show_debug(*value)
                elif kind == 'python':
                    show_python(*value)
                elif kind == 'result':
                    answer = value.answer if value.reason == 'finished' else f'本轮因 {value.reason} 结束，任务未确认完成。已执行的修改保留。'
                    finish_trial(job, value.reason, result=value)
                    finish_history(answer)
                    print('\n' + answer, flush=True)
                    debug_records(job)
                    job = None
                    prompt()
                elif kind == 'error':
                    usable = not isinstance(value, (WorkingStateLostError, ProcessCleanupError))
                    notice = '本轮失败：' + error_text(value, args.debug)
                    if not usable:
                        notice += '\n当前 Session 不可继续；/new 显式新建，不能恢复旧变量或重试未知调用。'
                    finish_trial(job, 'error', error=value)
                    finish_history(notice)
                    print(notice, flush=True)
                    debug_records(job)
                    job = None
                    prompt()
                continue
            text = value.strip()
            if experiment is not None and text in {'/new', '/clear-history'}:
                print('本次实验固定同一 Session 和历史策略；请 /quit 后用新日志开始另一批。', flush=True)
                prompt()
                continue
            if experiment is not None and (text == '/cap-ask' or text == QUESTION):
                if job:
                    print('上一 Turn 尚未结束。', flush=True)
                    continue
                try:
                    experiment.ready()
                except ValueError as error:
                    print(str(error), flush=True)
                    prompt()
                    continue
                text = QUESTION
            if text in {'/quit', '/new'}:
                pending = text[1:]
                if job:
                    print('正在停止当前 Turn…', flush=True)
                    job.cancelled = True
            elif text == '/stop':
                if job:
                    print('正在停止当前 Turn…', flush=True)
                    job.cancelled = True
                else:
                    print('没有活动任务。', flush=True)
                    prompt()
            elif text == '/help':
                print('/stop 停止 · /new 清空聊天与 Python 状态 · /quit 退出\n'
                      '/workspace Python 工作变量目录 · /status 状态与预算 · /history 聊天历史\n'
                      '/clear-history 只清聊天，保留 Python 状态。每行任务对应一个 Turn；忙时不排队。', flush=True)
                if job is None:
                    prompt()
            elif text == '/status':
                print(f'目录：{args.cwd}\nSession：{session.identity if session else "无"} · '
                      f'{"可用" if usable else "不可用"} · Turn：{session.active_turn if session else None}\n'
                      f'历史：{history.size}/{history.limit} 字节 · 单次上下文上限：{args.context_bytes} 字节', flush=True)
                if job is None:
                    prompt()
            elif job:
                print('任务进行中；请 /stop、/new 或 /quit。新任务未提交。', flush=True)
            elif text.startswith('/cap-'):
                try:
                    if experiment is None:
                        raise ValueError('请用 --cap-log 指定一个新的实验日志文件。')
                    parts = text.split()
                    if parts[0] == '/cap-env' and len(parts) == 2 and parts[1] in {'3', '4'}:
                        names = experiment.publish(int(parts[1]))
                        print('Host 已发布：' + ', '.join(names) + f'（{len(names)} 个，含 run）', flush=True)
                    elif parts[0] == '/cap-report' and len(parts) == 1:
                        print(json.dumps(experiment.report(), ensure_ascii=False, indent=2), flush=True)
                    elif parts[0] == '/cap-check' and len(parts) >= 2:
                        looked = 'unknown'
                        if '--look' in parts:
                            index = parts.index('--look')
                            if index != len(parts) - 2 or parts[-1] not in {'yes', 'no', 'unknown'}:
                                raise ValueError('--look yes|no|unknown 必须放在末尾。')
                            looked, parts = parts[-1], parts[:index]
                        count = int(parts[1])
                        if count < 0:
                            raise ValueError('总数不能为负数。')
                        print(json.dumps(experiment.check(count, parts[2:], looked), ensure_ascii=False), flush=True)
                    else:
                        raise ValueError('/cap-env 4|3；/cap-ask；/cap-check 数量 名称... [--look yes|no|unknown]；/cap-report')
                except (ValueError, NervusError) as error:
                    print('实验控制：' + str(error), flush=True)
                prompt()
            elif text == '/history':
                for message in history.messages:
                    print(f"{message['role']}: {message['content']}", flush=True)
                prompt()
            elif text == '/clear-history':
                history.clear()
                print('聊天历史已清空；Python 变量、函数和文件不变。', flush=True)
                prompt()
            elif text == '/workspace':
                if usable:
                    directory = session.inspect_namespace()
                    for entry in directory['entries']:
                        print(json.dumps(entry, ensure_ascii=False), flush=True)
                    print(f'目录元数据：{json.dumps({k:v for k,v in directory.items() if k != "entries"}, ensure_ascii=False)}', flush=True)
                else:
                    print('Session 不可用，请 /new。', flush=True)
                prompt()
            elif text.startswith('/'):
                print('未知命令；/help 查看。', flush=True)
                prompt()
            elif text:
                if experiment is not None and text != QUESTION:
                    print('实验模式只提交固定问题：/cap-ask。没有写入聊天历史。', flush=True)
                    prompt()
                    continue
                if not usable:
                    print('Session 不可用，请 /new。任务未提交。', flush=True)
                    prompt()
                    continue
                try:
                    prior = history.begin(text)
                except ContextBudgetError as error:
                    print(str(error), flush=True)
                    prompt()
                    continue
                try:
                    job = Job(session, factory(), text, prior, history.cleared, args.cwd, events, args)
                    if experiment is not None:
                        job.trial = experiment.begin()
                        if hasattr(job.model, 'trace'):
                            job.model.trace = job.trial.wire
                    job.thread.start()
                except Exception as error:
                    job = None
                    finish_history('任务未启动：' + error_text(error))
                    print('任务未启动：' + error_text(error, args.debug), flush=True)
                    prompt()
    except Exception as error:
        print('终端错误：' + error_text(error, args.debug), flush=True)
        exit_code = 1
    finally:
        try:
            if job is not None:
                job.cancelled = True
                deadline = time.monotonic() + 5
                while job.thread.is_alive():
                    try:
                        if job.stop() is not None:
                            break
                    except (WorkingStateLostError, ProcessCleanupError):
                        break  # close must still attempt the remaining cleanup.
                    if time.monotonic() >= deadline:
                        raise SessionStateError('任务启动/停止未确认，不能宣告退出完成')
                    time.sleep(0.01)
            if session is not None:
                session.close()
        except Exception as error:
            print('清理未确认：' + error_text(error, args.debug), flush=True)
            exit_code = 1
        else:
            if session is not None:
                print('Session 已关闭。', flush=True)
        finally:
            try:
                if experiment is not None:
                    experiment.write({'event': 'experiment_closed', 'report': experiment.report()})
                    experiment.stream.close()
            except OSError:
                print('实验日志关闭失败；不能宣称记录完整。', flush=True)
                exit_code = 1
            finally:
                signal.signal(signal.SIGINT, previous_handler)
    return exit_code


def positive(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return result


def positive_seconds(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError('must be positive and finite')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description='Nervus 终端 Agent：Python 编程、持久状态与受管程序。')
    parser.add_argument('--cwd', type=Path, default=Path.cwd())
    parser.add_argument('--env-file', type=Path)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--show-python', action=argparse.BooleanOptionalAction, default=False,
                        help='显示 Python 代码、执行结果及 stdout/stderr；默认隐藏，--debug 优先')
    parser.add_argument('--history-bytes', type=positive, default=32768)
    parser.add_argument('--context-bytes', type=positive, default=98304)
    parser.add_argument('--max-decisions', type=positive, default=8)
    parser.add_argument('--call-budget', type=positive, default=20)
    parser.add_argument('--max-tokens', type=positive, default=1536)
    parser.add_argument('--execution-timeout', type=positive_seconds, default=30)
    parser.add_argument('--stop-grace', type=positive_seconds, default=2)
    parser.add_argument('--cap-log', type=Path, help='记录手工 4→3→4 实验；新 JSONL 文件，不覆盖旧记录')
    parser.add_argument('--offline-script', type=Path, help='显式离线 JSON 动作脚本，仅用于验收；不请求 API')
    args = parser.parse_args(argv)
    args.cwd = args.cwd.resolve()
    if args.cap_log is not None:
        args.cap_log = args.cap_log.resolve()
        if args.cap_log.exists():
            parser.error('--cap-log must name a new file; existing records are never overwritten')
    if not args.cwd.is_dir():
        parser.error('--cwd must be an existing directory')
    try:
        if args.offline_script:
            factory = offline_factory(args.offline_script.resolve())
        else:
            if args.env_file:
                load_env(args.env_file.resolve())
            if not os.environ.get('DEEPSEEK_API_KEY', '').strip():
                parser.error('Set DEEPSEEK_API_KEY or provide --env-file')
            if os.environ.get('DEEPSEEK_MODEL', 'deepseek-v4-flash') != 'deepseek-v4-flash':
                parser.error('This terminal uses deepseek-v4-flash only')
            factory = lambda: DeepSeekFlash(max_tokens=args.max_tokens, debug_responses=args.debug,
                base_url=os.environ.get('DEEPSEEK_BASE_URL', 'https://api.deepseek.com'))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(f'Invalid startup configuration: {type(error).__name__}')
    with chdir(args.cwd):
        return interact(args, factory, queue.Queue())


if __name__ == '__main__':
    raise SystemExit(main())
