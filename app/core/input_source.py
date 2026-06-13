from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import List, Optional

import app.services.human_pose as human_pose


@dataclass(frozen=True)
class PointerSample:
    source: str
    x: float
    y: float
    timestamp: float


@dataclass(frozen=True)
class InputAction:
    source: str
    action: str
    timestamp: float


@dataclass(frozen=True)
class PointerClick:
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
        self._actions: List[InputAction] = []
        self._clicks: List[PointerClick] = []

    def submit_pointer(self, *, source: str, x: float, y: float, timestamp: Optional[float] = None) -> None:
        ts = time.monotonic() if timestamp is None else timestamp
        clamped_x = max(0.0, min(1.0, float(x)))
        clamped_y = max(0.0, min(1.0, float(y)))
        sample = PointerSample(source=source, x=clamped_x, y=clamped_y, timestamp=ts)
        with self._lock:
            self._latest_pointer_by_source[source] = sample

    def clear_pointer(self, source: str) -> None:
        with self._lock:
            self._latest_pointer_by_source.pop(source, None)

    def submit_action(self, *, source: str, action: str, timestamp: Optional[float] = None) -> None:
        ts = time.monotonic() if timestamp is None else timestamp
        with self._lock:
            self._actions.append(InputAction(source=source, action=action, timestamp=ts))

    def set_button_down(self, *, source: str, is_down: bool) -> None:
        with self._lock:
            self._button_down_by_source[source] = bool(is_down)

    def is_button_down(self, *, source: str) -> bool:
        with self._lock:
            return self._button_down_by_source.get(source, False)

    def submit_click(self, *, source: str, x: float, y: float, timestamp: Optional[float] = None) -> None:
        ts = time.monotonic() if timestamp is None else timestamp
        click = PointerClick(
            source=source,
            x=max(0.0, min(1.0, float(x))),
            y=max(0.0, min(1.0, float(y))),
            timestamp=ts,
        )
        with self._lock:
            self._clicks.append(click)

    def ingest_pose(self, pose_results) -> None:
        finger_x, finger_y = human_pose.get_right_index_finger_position(pose_results)
        if finger_x is None or finger_y is None:
            self.clear_pointer("pose")
            return
        self.submit_pointer(source="pose", x=finger_x, y=finger_y)

    def get_active_pointer(self, *, max_age_sec: float = 1.0) -> Optional[PointerSample]:
        now = time.monotonic()
        with self._lock:
            if not self._latest_pointer_by_source:
                return None
            newest = max(self._latest_pointer_by_source.values(), key=lambda sample: sample.timestamp)
            if now - newest.timestamp > max_age_sec:
                return None
            return newest

    def pop_actions(self, *, max_age_sec: float = 2.0) -> List[InputAction]:
        now = time.monotonic()
        with self._lock:
            actions = self._actions
            self._actions = []

        fresh_actions: List[InputAction] = []
        for action in actions:
            if now - action.timestamp <= max_age_sec:
                fresh_actions.append(action)
        return fresh_actions

    def pop_clicks(self, *, max_age_sec: float = 1.0) -> List[PointerClick]:
        now = time.monotonic()
        with self._lock:
            clicks = self._clicks
            self._clicks = []

        fresh_clicks: List[PointerClick] = []
        for click in clicks:
            if now - click.timestamp <= max_age_sec:
                fresh_clicks.append(click)
        return fresh_clicks
