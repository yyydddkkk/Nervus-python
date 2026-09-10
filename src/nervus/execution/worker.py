"""Own the namespace and run code against a fixed Turn snapshot."""

import ast
import asyncio
import inspect
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr

from .scope import TurnScope, origin, task_factory
from .output import CapturedStream, OutputBuffer
from .namespace import namespace_directory
from ..composition.capability import copy_data
from ..process import ProcessRunner, validate_argv
from ..composition.environment import Environment
from ..errors import CodeExecutionError, SessionStateError, CapabilityReleaseError


class _Reference:
    __slots__ = ("_worker", "_identity", "_interface_version")

    def __init__(self, worker, identity, interface_version):
        self._worker = worker
        self._identity = identity
        self._interface_version = interface_version

    def __call__(self, *args, **kwargs):
        source = self._worker.check_origin()
        return self._worker.invoke(self._identity, self._interface_version, source, args, kwargs)


# Native slot readers avoid user-defined attribute lookup during directory inspection.
_REFERENCE_SLOTS = tuple(_Reference.__dict__[name] for name in _Reference.__slots__)


class _Tools:
    def __init__(self, worker):
        self._worker = worker

    def __getattr__(self, name):
        scope, _ = self._worker.check_origin()
        identity = scope.snapshot.names.get(name)
        if identity is None:
            raise AttributeError(f"Capability unavailable: {name}")
        capability = scope.snapshot.bindings[identity].capability
        return _Reference(self._worker, identity, capability.interface_version)


class Worker:
    def __init__(self, session, facts, drain_timeout, output_limit, processes=None):
        self.session = session
        self.facts = facts
        self.processes = processes
        self.process_results = {}
        self.inspection_events = []
        self.drain_timeout = drain_timeout
        self.output_limit = output_limit
        self.last_output = None
        self.last_end = None
        self.environment = Environment()
        self.scope = None
        self.namespace = {"__name__": "__nervus_session__", "tools": _Tools(self)}
        self.namespace["inspect_workspace"] = self.inspect_namespace
        self._hidden = dict(self.namespace)
        self.turn_number = self.execution_number = self.call_number = 0

    def _reference_info(self, value):
        if type(value) is not _Reference:
            return None
        try:
            owner, identity, interface = (slot.__get__(value) for slot in _REFERENCE_SLOTS)
        except AttributeError:
            return {"callable": False, "availability": "invalid_reference"}
        if owner is not self or type(identity) is not str or type(interface) is not str:
            reason = "invalid_reference"
        elif self.scope is None:
            reason = "no_active_turn"
        elif self.scope.closing:
            reason = "turn_closed"
        elif identity not in self.scope.snapshot.bindings:
            reason = "unavailable"
        elif (type(self.scope.snapshot.bindings[identity].capability.interface_version) is not str
              or self.scope.snapshot.bindings[identity].capability.interface_version != interface):
            reason = "interface_mismatch"
        elif self.scope.remaining_calls <= 0:
            reason = "budget_exhausted"
        else:
            reason = "available"
        return {"callable": reason == "available", "availability": reason}

    def inspect_namespace(self, max_entries=50, max_bytes=8192, prefix=""):
        # __builtins__ is inserted by eval. Its ordinary builtins dictionary is
        # interpreter infrastructure; don't inspect or serialize it.
        hidden = {**self._hidden, "__builtins__": __builtins__}
        if type(__builtins__) is not dict:
            hidden["__builtins__"] = vars(__builtins__)
        result = namespace_directory(self.namespace, hidden=hidden, reference_info=self._reference_info,
                                     max_entries=max_entries, max_bytes=max_bytes, prefix=prefix)
        source = origin.get()
        if source is not None:
            self.inspection_events.append({"turn": source[0].number, "execution": source[1],
                                           "kind": "inspect_workspace", "completed": True})
        return result

    def check_origin(self):
        source = origin.get()
        if source is None or source[0] is not self.scope:
            raise SessionStateError("No active Turn for this execution")
        source[0].check()
        return source

    def confirm(self, fact):
        self.facts.send(fact)
        if not self.facts.recv():
            raise SessionStateError("Call fact rejected by supervisor")

    async def _process_result(self, call):
        while call not in self.process_results:
            if self.processes.poll():
                response = self.processes.recv()
                self.process_results[response["call"]] = response
            else:
                await asyncio.sleep(0.005)
        return self.process_results.pop(call)

    async def _run_process(self, fact, instance, argv):
        argv = validate_argv(argv)
        self.processes.send({"op": "start", "fact": fact, "argv": argv, "limit": instance.output_limit})
        try:
            response = await self._process_result(fact["call"])
        except asyncio.CancelledError:
            self.processes.send({"op": "cancel", "call": fact["call"]})
            await self._process_result(fact["call"])
            raise
        if response["cancelled"]:
            raise asyncio.CancelledError()
        if not response["ok"]:
            raise RuntimeError(response["error"])
        return response["result"]

    async def invoke(self, identity, interface, source, args, kwargs):
        scope, execution = self.check_origin()
        if source[0] is not scope:
            raise SessionStateError("A call request cannot cross Turns")
        execution = source[1]
        binding = scope.snapshot.bindings.get(identity)
        if binding is None:
            raise LookupError("Capability unavailable in this Turn")
        capability = binding.capability
        if capability.interface_version != interface:
            raise TypeError("Capability interface version mismatch")
        args, kwargs = copy_data(args), copy_data(kwargs)
        inspect.signature(binding.instance).bind(*args, **kwargs)
        if scope.remaining_calls <= 0:
            raise SessionStateError("Turn call budget exhausted")
        scope.remaining_calls -= 1
        self.call_number += 1
        fact = {"session": self.session, "generation": 1, "turn": scope.number,
                "execution": execution, "revision": scope.snapshot.revision,
                "call": self.call_number, "identity": identity,
                "interface_version": capability.interface_version,
                "implementation_version": capability.implementation_version}
        self.confirm({**fact, "status": "admitted"})
        try:
            scope.check()
            if type(binding.instance) is ProcessRunner:
                result = await self._run_process(fact, binding.instance, *args, **kwargs)
            else:
                result = copy_data(await binding.instance(*args, **kwargs))
        except asyncio.CancelledError:
            self.confirm({**fact, "status": "cancelled"})
            raise
        except Exception as error:
            self.confirm({**fact, "status": "failed", "error": f"{type(error).__name__}: {error}"})
            raise
        self.confirm({**fact, "status": "succeeded", "result": result})
        return result

    async def execute(self, code, exports):
        if self.scope is None:
            raise SessionStateError("begin_turn() must precede execute()")
        self.scope.check()
        self.execution_number += 1
        token = origin.set((self.scope, self.execution_number))
        try:
            filename = f"<nervus-code turn={self.scope.number} execution={self.execution_number}>"
            compiled = compile(code, filename, "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
            result = eval(compiled, self.namespace, self.namespace)
            if inspect.isawaitable(result):
                await result
            return {name: copy_data(self.namespace[name]) for name in exports}
        except asyncio.CancelledError as error:
            raise CodeExecutionError("CancelledError: code entry cancelled") from error
        except Exception as error:
            locations = [f"  at {frame.filename}:{frame.lineno} in {frame.name}"
                         for frame in traceback.extract_tb(error.__traceback__)
                         if frame.filename.startswith("<nervus-code ")]
            if isinstance(error, SyntaxError):
                locations.append(f"  at {error.filename}:{error.lineno}:{error.offset}")
            message = f"{type(error).__name__}: {error}"
            if len(message) > 2048:
                message = message[:2048] + " [exception text truncated]"
            raise CodeExecutionError("\n".join([message, *locations[-8:]])) from error
        finally:
            origin.reset(token)

    async def handle(self, command, arguments):
        target = arguments.pop("target_turn", None)
        if target is not None:
            actual = self.scope.number if self.scope is not None else None
            if command == "output" and actual is None and self.last_output is not None:
                actual = self.last_output["turn"]
            if actual != target:
                raise SessionStateError(f"Target Turn {target} is not current (current={actual})")
        if command == "publish":
            return await self.environment.publish(arguments["capabilities"],
                                                  self.scope.snapshot if self.scope else None)
        if command == "update":
            return await self.environment.update(**arguments,
                retained=self.scope.snapshot if self.scope else None)
        if command == "begin":
            if self.scope is not None:
                raise SessionStateError("The previous Turn has not ended")
            self.turn_number += 1
            self.scope = TurnScope(self.turn_number, self.environment.published, arguments["call_budget"],
                                   output=OutputBuffer(self.output_limit))
            return {"turn": self.scope.number, "revision": self.scope.snapshot.revision}
        if command == "execute":
            return await self.execute(**arguments)
        if command == "inspections":
            return self.inspection_events
        if command == "inspect_namespace":
            return self.inspect_namespace(**arguments)
        if command == "output":
            if self.scope is not None:
                return {"turn": self.scope.number, "last_execution": self.execution_number,
                        **self.scope.output.snapshot()}
            return self.last_output
        if command == "describe":
            if self.scope is None:
                raise SessionStateError("No active Turn")
            return {"turn": self.scope.number, "revision": self.scope.snapshot.revision,
                    "remaining_calls": self.scope.remaining_calls,
                    "capabilities": [
                        {"name": binding.capability.name,
                         "identity": binding.capability.identity,
                         "interface_version": binding.capability.interface_version,
                         "implementation_version": binding.capability.implementation_version,
                         "description": binding.capability.description,
                         "returns": binding.capability.returns,
                         "signature": str(inspect.signature(binding.instance))}
                        for binding in self.scope.snapshot.bindings.values()]}
        if command == "end":
            if self.scope is None:
                raise SessionStateError("No active Turn")
            report = await self.scope.drain(self.drain_timeout)
            self.last_output = {"turn": self.scope.number, "last_execution": self.execution_number,
                                **self.scope.output.snapshot()}
            self.last_end = {"turn": self.scope.number, "revision": self.scope.snapshot.revision,
                             "stopped": True, "working_state_lost": False,
                             "tasks": report, "output": self.last_output}
            self.scope = None
            try:
                await self.environment.retire(None)
            except CapabilityReleaseError as error:
                self.last_end["release_errors"] = [str(error)]
                raise
            return report
        if command == "close":
            failures = []
            if self.scope is not None:
                try:
                    await self.handle("end", {})
                except CapabilityReleaseError as error:
                    # Tasks have stopped; retirement failure must not skip the
                    # published instances. Drain failure still escalates to kill.
                    failures.append(str(error))
            try:
                await self.environment.close()
            except CapabilityReleaseError as error:
                failures.append(str(error))
            if failures:
                raise CapabilityReleaseError("; ".join(failures))
            return None
        raise SessionStateError(f"Unknown internal command: {command}")


async def _serve(commands, facts, session, drain_timeout, output_limit, stops, processes):
    worker = Worker(session, facts, drain_timeout, output_limit, processes)
    asyncio.get_running_loop().set_task_factory(task_factory)
    active = None
    active_command = None
    stopping = None

    async def stop_turn(turn):
        scope = worker.scope
        if scope is None:
            if worker.last_end is not None and worker.last_end["turn"] == turn:
                return worker.last_end
            raise SessionStateError("Target Turn is not active")
        if scope.number != turn:
            raise SessionStateError("Target Turn is not active")
        scope.closing = True
        if active is not None and not active.done():
            if active_command == "execute":
                active.cancel()
            await asyncio.gather(active, return_exceptions=True)
        if worker.scope is scope:
            try:
                await worker.handle("end", {"target_turn": turn})
            except CapabilityReleaseError as error:
                # Execution has stopped; report resource failure without hiding
                # successful drain or losing the still-live namespace.
                worker.last_end["release_errors"] = [str(error)]
        return worker.last_end

    async def listen_stops():
        nonlocal stopping
        while True:
            try:
                # A tiny control message; poll cooperatively so shutdown never
                # strands an executor thread in recv().
                while not stops.poll():
                    await asyncio.sleep(0.005)
                turn = stops.recv()
            except (EOFError, OSError):
                return
            stopping = asyncio.create_task(stop_turn(turn))
            try:
                result = await stopping
                stops.send({"ok": True, "result": result})
            except Exception as error:
                stops.send({"ok": False, "error": type(error).__name__, "message": str(error)})
            finally:
                stopping = None

    listener = asyncio.create_task(listen_stops())
    try:
        while True:
            command, arguments = await asyncio.to_thread(commands.recv)
            if stopping is not None:
                await asyncio.gather(stopping, return_exceptions=True)
            active_command = command
            active = asyncio.create_task(worker.handle(command, arguments))
            try:
                result = await active
                commands.send({"ok": True, "result": result})
            except asyncio.CancelledError:
                commands.send({"ok": False, "error": "CodeExecutionError", "message": "Code entry cancelled by stop"})
            except Exception as error:
                commands.send({"ok": False, "error": type(error).__name__, "message": str(error)})
            if command == "close":
                return
    finally:
        listener.cancel()
        await asyncio.gather(listener, return_exceptions=True)


def run_worker(commands, facts, session, drain_timeout, output_limit, stops, processes):
    try:
        with redirect_stdout(CapturedStream("stdout", sys.stdout)), redirect_stderr(CapturedStream("stderr", sys.stderr)):
            asyncio.run(_serve(commands, facts, session, drain_timeout, output_limit, stops, processes))
    finally:
        commands.close()
        facts.close()
        stops.close()
        processes.close()
