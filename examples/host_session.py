"""Thin Host walkthrough: uv run --locked python examples/host_session.py.

Only ScriptedModel and the project Python interpreter; no network or model API.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial
import sys
import threading
import time

from nervus import Capability, Code, Finish, ProcessRunner, ScriptedModel, Session
from basic_session import TaggedEcho


def main():
    read = Capability('read', partial(TaggedEcho, 'old'), implementation_version='old',
                      description='Return a tagged value and instance call count.',
                      returns='dict: marker, value, calls')
    updated = replace(read, factory=partial(TaggedEcho, 'new'), implementation_version='new')
    process = Capability('run', partial(ProcessRunner, output_limit=48),
                         description='Run argv without Shell parsing; wait for process exit.',
                         returns='dict: returncode, stdout, stderr, stdout_truncated, stderr_truncated')
    argv = [sys.executable, '-c',
            "import sys; print('total=' + sys.argv[1]); print('x'*80); print('diagnostic', file=sys.stderr)"]

    with Session() as session:
        session.publish([read, process])  # publish always describes the complete environment.

        def publish_during_turn(context):
            output = context.feedback[-1].values['output']
            assert output['returncode'] == 0 and output['stdout_truncated']
            print('Process stdout:', repr(output['stdout']), '(truncated)')
            print('Process stderr:', repr(output['stderr']))
            # This callback is Host test scaffolding, not a model-accessible publish tool.
            session.publish([updated, process])
            return Code('during = await reader(total())', exports=('during',))

        def finish_first(context):
            assert context.feedback[-1].values['during']['marker'] == 'old'
            return Finish('Turn 1: published revision 2, still using the old snapshot.')

        first = ScriptedModel([
            Code('saved = [20, 22]\ndef total(): return sum(saved)\n'
                 'reader = tools.read\nrunner = tools.run\n'
                 f'output = await runner({argv!r} + [str(total())])', exports=('output',)),
            publish_during_turn, finish_first,
        ])
        print(session.run('Save work, run a program, and observe a capability update.', first).answer)

        # The Host owns its thread. A synchronous model call need not return for
        # stop to revoke this Turn's execution rights and drain its worker tasks.
        deciding, release_model = threading.Event(), threading.Event()

        def pending_decision(context):
            current = context.feedback[-1].values['current']
            assert current['marker'] == 'new' and current['value'] == 42
            assert context.revision == 2
            print('Turn 2: reused total() and reader; marker =', current['marker'])
            deciding.set()
            if not release_model.wait(10):
                raise TimeoutError('Host did not release the scripted model')
            return Finish('This late answer must be discarded.')

        long_argv = [sys.executable, '-c', 'import time; time.sleep(60)']
        second = ScriptedModel([
            Code('import asyncio\ncurrent = await reader(total())\n'
                 f'pending = asyncio.create_task(runner({long_argv!r}))', exports=('current',)),
            pending_decision,
        ])
        with ThreadPoolExecutor(max_workers=1) as executor:
            running = executor.submit(session.run, 'Start work that the Host will stop.', second)
            try:
                if not deciding.wait(5):
                    raise TimeoutError('Scripted model did not reach the decision barrier')
                turn = session.active_turn  # Capture this target; never retry against a newer Turn.
                deadline = time.monotonic() + 5
                while not any(e['event'] == 'running' and e['turn'] == turn
                              for e in session.process_events):
                    if time.monotonic() >= deadline:
                        raise TimeoutError('External program did not start')
                    time.sleep(0.01)  # Observe real supervisor facts, not an assumed startup delay.
                stopped = session.stop(turn, grace=2)
                assert stopped['stopped'] and not stopped['working_state_lost']
                assert not running.done()  # The model itself is still waiting.
                print('Stopped Turn', turn, ': worker tasks and external process reclaimed; state retained.')
            finally:
                release_model.set()
            result = running.result(timeout=5)
            assert result.reason == 'stopped' and result.answer is None

        third = ScriptedModel([
            Code("assert pending.done() and pending.cancelled()\n"
                 'current = await reader(total())', exports=('current',)),
            lambda context: Finish(f"Turn 3: state retained, total = {context.feedback[-1].values['current']['value']}."),
        ])
        print(session.run('Continue after cooperative stop.', third).answer)
        print('Calls:', [(c['turn'], c['implementation_version'], c['status']) for c in session.calls])
        print('Process exits:', [(e['turn'], e['returncode']) for e in session.process_events if e['event'] == 'exited'])
    print('Session closed.')


if __name__ == '__main__':
    main()
