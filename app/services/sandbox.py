"""Safely run LLM-generated animation code on the flip-dot display.

Generated code never touches the display, filesystem or network. It only
implements a pure *frame generator* with this contract::

    def setup(width, height):
        '''Return the initial state (any object; stays inside the worker).'''

    def step(state, t, width, height):
        '''Return (new_state, frame) OR just frame. frame is a (height, width)
        array of 0/1 values; t is elapsed seconds since the script started.'''

Three layers keep this safe enough to run without a human reading the code:

1. ``validate_source`` — a static AST allow-list rejects imports outside
   ``numpy``/``math``/``random``, dunder access (the usual sandbox-escape
   route), and dangerous builtins, *before* anything runs.
2. The code executes in a **dedicated subprocess** with a restricted
   ``__builtins__`` and ``numpy``/``math``/``random`` pre-injected. The worker
   is launched via ``python -c`` (not ``multiprocessing``) so it imports *only*
   numpy and this module — never the host app's heavy stack (mediapipe, cv2,
   serial). That matters: ``multiprocessing`` re-imports ``__main__`` to find
   the target, which would drag ~1 GB of the app's libraries into the worker
   and make the memory ``rlimit`` clamp below the already-consumed address
   space, so every allocation in an otherwise-fine script would ``MemoryError``.
3. The worker has a memory ``rlimit`` and every frame is fetched with a
   timeout — a hang, crash or OOM kills the child without touching the
   single-threaded display loop. Only a shape-checked ``uint8`` pixel buffer
   ever crosses back to the host; no untrusted object re-enters the parent.
"""

from __future__ import annotations

import ast
import os
import socket
import subprocess
import sys
import traceback
from multiprocessing.connection import Connection
from pathlib import Path

import numpy as np

# Repo root, so the worker subprocess can import ``app.services.sandbox``.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])


# Modules generated code may use. Everything else is rejected statically.
ALLOWED_MODULES = {"numpy", "math", "random"}

# Builtins that are IO / introspection / code-execution vectors. Even though
# the worker namespace omits them, rejecting them in the AST gives Claude a
# clear, early error to self-correct against.
BLOCKED_NAMES = {
    "eval",
    "exec",
    "compile",
    "open",
    "__import__",
    "input",
    "breakpoint",
    "exit",
    "quit",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
    "memoryview",
    "help",
}

# Builtins the worker namespace *does* expose. Deliberately omits open/eval/
# exec/__import__/getattr/setattr and other escape vectors.
_SAFE_BUILTINS = {
    name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
    for name in (
        "abs",
        "all",
        "any",
        "bool",
        "bytes",
        "bytearray",
        "complex",
        "dict",
        "divmod",
        "enumerate",
        "filter",
        "float",
        "frozenset",
        "int",
        "len",
        "list",
        "map",
        "max",
        "min",
        "pow",
        "print",
        "range",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
        "True",
        "False",
        "None",
        "abs",
        "bin",
        "hex",
        "oct",
        "ord",
        "chr",
        "isinstance",
        "issubclass",
    )
    if (name in __builtins__ if isinstance(__builtins__, dict) else hasattr(__builtins__, name))
}

# Defaults (overridable via env for constrained hosts like a Pi).
DEFAULT_MEM_LIMIT_MB = int(os.getenv("SANDBOX_MEM_MB", "1024"))
DEFAULT_FRAME_TIMEOUT = float(os.getenv("SANDBOX_FRAME_TIMEOUT", "0.25"))
DEFAULT_STARTUP_TIMEOUT = float(os.getenv("SANDBOX_STARTUP_TIMEOUT", "8.0"))


class ScriptValidationError(ValueError):
    """Raised when generated source fails the static safety checks."""


class SandboxStartupError(RuntimeError):
    """Raised when the worker process fails to start the script."""


# --- Static validation ----------------------------------------------------


class _Validator(ast.NodeVisitor):
    def __init__(self) -> None:
        self.errors: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root not in ALLOWED_MODULES:
                self.errors.append(f"import of '{alias.name}' is not allowed")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        root = (node.module or "").split(".")[0]
        if root not in ALLOWED_MODULES:
            self.errors.append(f"import from '{node.module}' is not allowed")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.attr, str) and node.attr.startswith("__"):
            self.errors.append(f"access to dunder attribute '{node.attr}' is not allowed")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("__"):
            self.errors.append(f"use of dunder name '{node.id}' is not allowed")
        elif node.id in BLOCKED_NAMES:
            self.errors.append(f"use of '{node.id}' is not allowed")
        self.generic_visit(node)


def validate_source(code: str) -> None:
    """Raise ``ScriptValidationError`` if ``code`` is unsafe or malformed."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ScriptValidationError(f"syntax error: {exc}") from exc

    validator = _Validator()
    validator.visit(tree)
    if validator.errors:
        raise ScriptValidationError("; ".join(sorted(set(validator.errors))))

    func_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    if "step" not in func_names:
        raise ScriptValidationError("script must define a step(state, t, width, height) function")


# --- Worker process --------------------------------------------------------


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """A drop-in ``__import__`` that only resolves allow-listed modules."""
    root = name.split(".")[0]
    if root not in ALLOWED_MODULES:
        raise ImportError(f"import of '{name}' is not allowed in sandboxed scripts")
    return __import__(name, globals, locals, fromlist, level)


def _build_namespace() -> dict:
    import math
    import random

    builtins = dict(_SAFE_BUILTINS)
    builtins["__import__"] = _safe_import
    return {
        "__builtins__": builtins,
        "np": np,
        "numpy": np,
        "math": math,
        "random": random,
    }


def _apply_rlimits(mem_limit_bytes: int) -> None:
    try:
        import resource
    except ImportError:  # pragma: no cover - non-Unix
        return
    if mem_limit_bytes > 0:
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mem_limit_bytes, mem_limit_bytes))
        except (ValueError, OSError):  # pragma: no cover - host dependent
            pass


def _short_tb() -> str:
    return traceback.format_exc(limit=3).strip().splitlines()[-1]


def _worker_main(conn, code: str, width: int, height: int, mem_limit_bytes: int) -> None:
    """Child-process entry point. Loads the script, then serves frame requests."""
    try:
        _apply_rlimits(mem_limit_bytes)
        namespace = _build_namespace()
        exec(compile(code, "<script>", "exec"), namespace)  # noqa: S102 - sandboxed
        setup = namespace.get("setup")
        step = namespace.get("step")
        if not callable(step):
            conn.send(("error", "script must define a step(state, t, width, height) function"))
            return
        state = setup(width, height) if callable(setup) else None
        conn.send(("ready",))
    except Exception:  # noqa: BLE001 - report any load/setup failure to the host
        conn.send(("error", _short_tb()))
        return

    while True:
        try:
            msg = conn.recv()
        except (EOFError, KeyboardInterrupt):
            break
        if not msg or msg[0] == "stop":
            break
        if msg[0] != "step":
            continue
        try:
            result = step(state, msg[1], width, height)
            if isinstance(result, tuple) and len(result) == 2:
                state, frame = result
            else:
                frame = result
            frame = np.asarray(frame)
            if frame.shape != (height, width):
                conn.send(("error", f"frame shape {frame.shape} must be ({height}, {width})"))
                break
            frame = np.where(frame != 0, 1, 0).astype(np.uint8)
            conn.send(("frame", frame.tobytes()))
        except Exception:  # noqa: BLE001 - surface runtime errors to the host
            conn.send(("error", _short_tb()))
            break


def _worker_entry(fd: int) -> None:
    """Subprocess entry point (run via ``python -c``).

    Wraps the inherited socket ``fd`` as a Connection, reads the one-shot
    ``config`` message, then hands off to :func:`_worker_main`. Because the
    worker is launched this way, it imports only numpy and this module — none of
    the host application's heavy libraries.
    """
    conn = Connection(fd)
    try:
        msg = conn.recv()
    except (EOFError, OSError):
        return
    if not msg or msg[0] != "config":
        return
    _, code, width, height, mem_limit_bytes = msg
    _worker_main(conn, code, width, height, mem_limit_bytes)


class SandboxedScript:
    """A running, isolated frame generator. ``get_frame`` is fail-safe."""

    def __init__(
        self,
        code: str,
        width: int,
        height: int,
        *,
        mem_limit_mb: int = DEFAULT_MEM_LIMIT_MB,
        frame_timeout: float = DEFAULT_FRAME_TIMEOUT,
        startup_timeout: float = DEFAULT_STARTUP_TIMEOUT,
    ) -> None:
        validate_source(code)  # raises ScriptValidationError on unsafe code
        self.code = code
        self.width = width
        self.height = height
        self._mem_limit_bytes = max(0, mem_limit_mb) * 1024 * 1024
        self._frame_timeout = frame_timeout
        self._startup_timeout = startup_timeout
        self._proc: subprocess.Popen | None = None
        self._conn: Connection | None = None
        self._failed = False
        self._error: str | None = None

    @property
    def failed(self) -> bool:
        """Whether the script has failed and can no longer produce frames."""
        return self._failed

    @property
    def error(self) -> str | None:
        """The failure message, if the script has failed; otherwise None."""
        return self._error

    def start(self) -> None:
        """Launch the isolated worker process and wait for it to be ready."""
        # A socketpair carries the multiprocessing Connection protocol; one end
        # is inherited by a minimal `python -c` worker that imports only numpy.
        parent_sock, child_sock = socket.socketpair()
        child_fd = child_sock.fileno()
        os.set_inheritable(child_fd, True)
        bootstrap = f"from app.services.sandbox import _worker_entry; _worker_entry({child_fd})"
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(p for p in (_REPO_ROOT, env.get("PYTHONPATH", "")) if p)
        try:
            self._proc = subprocess.Popen(
                [sys.executable, "-c", bootstrap],
                pass_fds=(child_fd,),
                cwd=_REPO_ROOT,
                env=env,
            )
        finally:
            child_sock.close()  # the parent keeps only its own end of the pair
        self._conn = Connection(parent_sock.detach())

        try:
            self._conn.send(("config", self.code, self.width, self.height, self._mem_limit_bytes))
        except (OSError, BrokenPipeError) as exc:
            self._fail(f"failed to start worker: {exc}")
            raise SandboxStartupError(self._error or "failed to start worker") from exc

        if not self._conn.poll(self._startup_timeout):
            self._fail("script did not start in time")
            raise SandboxStartupError(self._error or "startup timeout")
        try:
            msg = self._conn.recv()
        except (EOFError, OSError) as exc:
            self._fail("worker died during startup")
            raise SandboxStartupError(self._error or "worker died") from exc
        if not msg or msg[0] != "ready":
            err = msg[1] if msg and len(msg) > 1 else "unknown startup error"
            self._fail(err)
            raise SandboxStartupError(err)

    def _alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def get_frame(self, t: float) -> np.ndarray | None:
        """Return the next frame, or ``None`` (and mark failed) on any problem.

        ``t`` is elapsed seconds since the script started.
        """
        if self._failed:
            return None
        if not self._alive():
            self._fail("worker process is not running")
            return None
        conn = self._conn
        if conn is None:
            self._fail("worker pipe closed")
            return None
        try:
            conn.send(("step", float(t)))
        except (BrokenPipeError, OSError):
            self._fail("worker pipe closed")
            return None
        if not conn.poll(self._frame_timeout):
            self._fail(f"frame timed out after {self._frame_timeout}s")
            return None
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            self._fail("worker died")
            return None
        kind = msg[0] if msg else None
        if kind == "frame":
            return np.frombuffer(msg[1], dtype=np.uint8).reshape(self.height, self.width).copy()
        if kind == "error":
            self._fail(msg[1])
            return None
        self._fail(f"unexpected worker message: {kind!r}")
        return None

    def stop(self) -> None:
        """Signal the worker to stop and terminate the process if needed."""
        proc = self._proc
        if proc is not None and proc.poll() is None and self._conn is not None:
            try:
                self._conn.send(("stop",))
            except (BrokenPipeError, OSError):
                pass
            try:
                proc.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=0.5)
                except subprocess.TimeoutExpired:  # pragma: no cover - last resort
                    proc.kill()
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:
                pass
        self._conn = None
        self._proc = None

    def _fail(self, message: str) -> None:
        self._failed = True
        self._error = message
        self.stop()
