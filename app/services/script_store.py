"""Named, on-disk library of generated animation scripts.

Mirrors the saved-board convention (``state/boards`` -> ``state/scripts``): each
animation is one ``<name>.py`` file holding its validated source, so a script
can be re-run later by name without regenerating it.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

_DEFAULT_DIR = Path(__file__).resolve().parents[2] / "state" / "scripts"
_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class ScriptStore:
    def __init__(self, scripts_dir: Optional[Path] = None) -> None:
        self._dir = Path(os.getenv("SCRIPT_STATES_DIR", str(scripts_dir or _DEFAULT_DIR)))

    @staticmethod
    def sanitize_name(name: str) -> str:
        candidate = str(name or "").strip()
        if not _NAME_RE.match(candidate):
            raise ValueError("name must be 1-64 chars of letters, digits, '-' or '_'")
        return candidate

    def _path(self, name: str) -> Path:
        return self._dir / f"{self.sanitize_name(name)}.py"

    def save(self, name: str, code: str) -> str:
        path = self._path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".py.tmp")
        tmp.write_text(code, encoding="utf-8")
        os.replace(tmp, path)
        return path.stem

    def load(self, name: str) -> Optional[str]:
        path = self._path(name)
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def list_names(self) -> list[str]:
        if not self._dir.exists():
            return []
        return sorted(entry.stem for entry in self._dir.glob("*.py"))

    def delete(self, name: str) -> bool:
        path = self._path(name)
        if not path.exists():
            return False
        path.unlink()
        return True
