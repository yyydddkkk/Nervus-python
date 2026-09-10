# Nervus Python

**English** | [简体中文](README.zh-CN.md)

**0.1.0a1 · Developer preview · Not yet published to PyPI.**

A code-driven Python Agent Kernel with an embeddable Host API and a terminal agent. Python variables, functions, and results persist within the same Session. Public APIs may still change.

Supports **Linux / CPython 3.14.x** only, with no runtime dependencies beyond the standard library. Windows, macOS, and other Python versions have not been validated.

## Installation and examples

From the source directory:

```sh
uv sync --locked
uv run --locked python examples/basic_session.py
uv run --locked python examples/scripted_session.py
uv run --locked python examples/host_session.py
uv run --locked python examples/partial_update.py
```

Examples: [Basic Session](examples/basic_session.py), [Scripted model](examples/scripted_session.py), [Host and managed processes](examples/host_session.py), and [Partial capability updates](examples/partial_update.py). None of these four examples calls a model API. The Host example starts and stops local Python subprocesses.

## Terminal agent

```sh
uv run --locked nervus --cwd /absolute/project --env-file /absolute/path/to/.env --show-python
```

Alternatively, set `DEEPSEEK_API_KEY` and omit `--env-file`. The target project's `.env` is not loaded automatically. The default model is `deepseek-v4-flash`. Submitting a task calls the model API and may incur charges.

- `/stop` or Ctrl+C: stop the current Turn.
- `/workspace`: inspect workspace metadata while idle.
- `/history` and `/status`: inspect conversation history and status; `/clear-history` clears only the conversation history.
- `/new`: close the old Session and clear conversation history and variables. Files already written are not deleted.
- `/quit` or EOF: clean up and exit.
- `--show-python`: display code and execution feedback; `--debug`: display full diagnostic details, which may contain sensitive information.

Each input line starts one Turn. Inputs are not queued while busy. Conversation history and model context have size limits; exceeding them pauses execution rather than silently truncating content. Conversation history is not retained across terminal restarts.

## Basic Host usage

```python
from nervus import Session, Code, Finish, ScriptedModel


def main():
    with Session() as session:
        result = session.run('Compute and save', ScriptedModel([
            Code('saved = 6 * 7', exports=('saved',)),
            Finish('Done'),
        ]))
        print(result.feedback[0].values)  # {'saved': 42}


if __name__ == '__main__':
    main()
```

Workers use the spawn start method. Host scripts must guard their entry point, and capability factories must be importable by spawned workers. See the [Host example](examples/host_session.py) and [partial update example](examples/partial_update.py) for capability registration and updates.

## Limitations and safety

- **Run trusted code only. This is not a sandbox.** Applications are responsible for isolating file, network, and other host-side effects.
- Code errors do not roll back variables or files. Worker loss does not trigger automatic recovery; use `/new` explicitly in the terminal.
- Stopping cannot forcibly cancel arbitrary synchronous model calls or manage arbitrary user-created threads and processes started outside the managed interface.
- Managed processes use POSIX process groups. Programs that detach from their group are not covered, and cleanup after a supervisor crash is not guaranteed.
- This project makes no guarantees about model task success rates or production reliability.

## Validation and contributing

```sh
uv run --locked --offline python -m unittest discover -s tests -t . -v
uv build --offline
```

Run `uv sync --locked` first to prepare the environment. The current test baseline is 102 tests, with no model API key required. CI runs this suite and builds the wheel and sdist on Linux / Python 3.14. uv's offline mode is not network isolation.

- [Contributing](CONTRIBUTING.md)
- [Security and vulnerability reporting](SECURITY.md)
- [Changelog](CHANGELOG.md)

Licensed under the [MIT License](LICENSE).
