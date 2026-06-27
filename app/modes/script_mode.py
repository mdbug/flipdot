"""Display mode that hosts an LLM-generated, sandboxed frame generator.

The heavy lifting (validation, isolation, resource limits) lives in
``app.services.sandbox``; this class is the thin glue that the render loop and
the MCP tools talk to. ``get_frame`` is called once per tick by the registry and
never raises — a failed/crashed script renders a small error frame instead of
disturbing the loop.
"""

from __future__ import annotations

import time

import numpy as np

import app.services.text as text
from app.services.sandbox import SandboxedScript
from app.services.script_store import ScriptStore


class ScriptMode:
    """Render-loop glue around a sandboxed, LLM-generated frame generator."""

    def __init__(self, width: int, height: int, store: ScriptStore | None = None) -> None:
        self.width = width
        self.height = height
        self._store = store or ScriptStore()
        self._script: SandboxedScript | None = None
        self._name: str = ""
        self._start_time = 0.0

    # --- render loop -----------------------------------------------------

    def get_frame(self, *_args: object, **_kwargs: object) -> np.ndarray:
        """Return the script's frame for the current elapsed time; never raises."""
        if self._script is None:
            return self._blank()
        # Pass elapsed wall-clock seconds so animation speed is independent of
        # the (variable) render frame rate.
        t = time.monotonic() - self._start_time
        frame = self._script.get_frame(t)
        if frame is None:
            return self._error_frame(self._script.error or "script stopped")
        return frame

    # --- control (called from MCP tools) ---------------------------------

    def run_script(self, code: str, name: str = "") -> dict:
        """Validate, isolate and start ``code``. Raises on unsafe/broken code."""
        script = SandboxedScript(code, self.width, self.height)  # validates
        script.start()  # spawns worker, runs setup; raises on startup failure
        self._replace(script, name)
        saved = ""
        if name:
            saved = self._store.save(name, code)
            self._name = saved
        return {"running": True, "name": saved}

    def stop_script(self) -> bool:
        """Stop the running script; return False if none was active."""
        if self._script is None:
            return False
        self._replace(None, "")
        return True

    def save_script(self, name: str) -> str:
        """Persist the running script's code under ``name``; return the saved name."""
        if self._script is None:
            raise ValueError("no script is running to save")
        saved = self._store.save(name, self._script.code)
        self._name = saved
        return saved

    def load_script(self, name: str) -> dict:
        """Load ``name`` from the store and start it; raise if it does not exist."""
        code = self._store.load(name)
        if code is None:
            raise ValueError(f"script '{name}' not found")
        return self.run_script(code, name=ScriptStore.sanitize_name(name))

    def list_scripts(self) -> dict:
        """Return stored script names and the active one."""
        return {"scripts": self._store.list_names(), "active": self._name}

    def get_code(self, name: str) -> str | None:
        """Return the source code of the saved script ``name``, or None if absent."""
        return self._store.load(name)

    def delete_script(self, name: str) -> bool:
        """Delete a stored script; return whether it existed."""
        return self._store.delete(name)

    def status(self) -> dict:
        """Return the active script name, whether it is running, and any error."""
        running = self._script is not None and not self._script.failed
        return {
            "name": self._name,
            "running": running,
            "error": self._script.error if self._script is not None else None,
        }

    # --- internals -------------------------------------------------------

    def _replace(self, script: SandboxedScript | None, name: str) -> None:
        if self._script is not None:
            self._script.stop()
        self._script = script
        self._name = name
        self._start_time = time.monotonic()

    def _blank(self) -> np.ndarray:
        return np.zeros((self.height, self.width), dtype=np.uint8)

    def _error_frame(self, message: str) -> np.ndarray:
        frame = self._blank()
        text.write_centered(frame, "SCRIPT", y=8, size=5)
        text.write_centered(frame, "ERROR", y=16, size=5)
        return frame
