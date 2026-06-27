from __future__ import annotations

import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import app.services.human_pose as human_pose


@dataclass(frozen=True)
class PointerSample:
    """A normalized (0..1) pointer position reported by an input source."""

    source: str
    x: float
    y: float
    timestamp: float


@dataclass(frozen=True)
class InputAction:
    """A named action (e.g. a button press) reported by an input source."""

    source: str
    action: str
    timestamp: float


@dataclass(frozen=True)
class PointerClick:
    """A discrete click at a normalized (0..1) position from an input source."""

    source: str
    x: float
    y: float
    timestamp: float


class InputHub:
    """Collect pointer/actions from multiple sources with last-input-wins reads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest_pointer_by_source: dict[str, PointerSample] = {}
        self._button_down_by_source: dict[str, bool] = {}
        self._actions: list[InputAction] = []
        self._clicks: list[PointerClick] = []

    def submit_pointer(
        self, *, source: str, x: float, y: float, timestamp: float | None = None
    ) -> None:
        """Record the latest pointer position for ``source`` (coords clamped to 0..1)."""
        ts = time.monotonic() if timestamp is None else timestamp
        clamped_x = max(0.0, min(1.0, float(x)))
        clamped_y = max(0.0, min(1.0, float(y)))
        sample = PointerSample(source=source, x=clamped_x, y=clamped_y, timestamp=ts)
        with self._lock:
            self._latest_pointer_by_source[source] = sample

    def clear_pointer(self, source: str) -> None:
        """Forget the last pointer position reported by ``source``."""
        with self._lock:
            self._latest_pointer_by_source.pop(source, None)

    def submit_action(self, *, source: str, action: str, timestamp: float | None = None) -> None:
        """Queue a named action from ``source`` for the next ``pop_actions`` read."""
        ts = time.monotonic() if timestamp is None else timestamp
        with self._lock:
            self._actions.append(InputAction(source=source, action=action, timestamp=ts))

    def set_button_down(self, *, source: str, is_down: bool) -> None:
        """Set the held/released state of ``source``'s primary button."""
        with self._lock:
            self._button_down_by_source[source] = bool(is_down)

    def is_button_down(self, *, source: str) -> bool:
        """Return whether ``source``'s primary button is currently held."""
        with self._lock:
            return self._button_down_by_source.get(source, False)

    def submit_click(
        self, *, source: str, x: float, y: float, timestamp: float | None = None
    ) -> None:
        """Queue a discrete click from ``source`` for the next ``pop_clicks`` read."""
        ts = time.monotonic() if timestamp is None else timestamp
        click = PointerClick(
            source=source,
            x=max(0.0, min(1.0, float(x))),
            y=max(0.0, min(1.0, float(y))),
            timestamp=ts,
        )
        with self._lock:
            self._clicks.append(click)

    def ingest_pose(self, pose_results: Any) -> None:
        """Update the ``"pose"`` pointer from the right index finger, or clear it if absent."""
        finger_x, finger_y = human_pose.get_right_index_finger_position(pose_results)
        if finger_x is None or finger_y is None:
            self.clear_pointer("pose")
            return
        self.submit_pointer(source="pose", x=finger_x, y=finger_y)

    def get_active_pointer(
        self,
        *,
        max_age_sec: float = 1.0,
        allowed_sources: Iterable[str] | None = None,
    ) -> PointerSample | None:
        """Return the newest fresh pointer (within ``max_age_sec``), optionally source-filtered."""
        now = time.monotonic()
        allowed = set(allowed_sources) if allowed_sources is not None else None
        with self._lock:
            if not self._latest_pointer_by_source:
                return None
            samples = list(self._latest_pointer_by_source.values())
            if allowed is not None:
                samples = [sample for sample in samples if sample.source in allowed]
            if not samples:
                return None
            newest = max(samples, key=lambda sample: sample.timestamp)
            if now - newest.timestamp > max_age_sec:
                return None
            return newest

    def pop_actions(
        self,
        *,
        max_age_sec: float = 2.0,
        allowed_sources: Iterable[str] | None = None,
    ) -> list[InputAction]:
        """Drain and return actions newer than ``max_age_sec``, optionally source-filtered."""
        now = time.monotonic()
        allowed = set(allowed_sources) if allowed_sources is not None else None
        with self._lock:
            actions = self._actions
            self._actions = []

        fresh_actions: list[InputAction] = []
        for action in actions:
            if now - action.timestamp <= max_age_sec:
                if allowed is not None and action.source not in allowed:
                    continue
                fresh_actions.append(action)
        return fresh_actions

    def pop_clicks(
        self,
        *,
        max_age_sec: float = 1.0,
        allowed_sources: Iterable[str] | None = None,
    ) -> list[PointerClick]:
        """Drain and return clicks newer than ``max_age_sec``, optionally source-filtered."""
        now = time.monotonic()
        allowed = set(allowed_sources) if allowed_sources is not None else None
        with self._lock:
            clicks = self._clicks
            self._clicks = []

        fresh_clicks: list[PointerClick] = []
        for click in clicks:
            if now - click.timestamp <= max_age_sec:
                if allowed is not None and click.source not in allowed:
                    continue
                fresh_clicks.append(click)
        return fresh_clicks
