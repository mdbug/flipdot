import logging
import re
import threading
import time
from typing import Any

import numpy as np

from app.core.mode_manager import ModeManager
from app.modes.contracts import Frame
from app.services.text import center_x, supported_characters, width, write, write_centered
from app.services.worldcup import get_worldcup_scorecard

logger = logging.getLogger(__name__)


class WorldCup:
    """Display live World Cup scores, polling ESPN and flashing on goals.

    An information mode (no pose interaction): it refreshes a scorecard on
    a background thread, renders the selected match, and plays a brief goal
    animation and score flash whenever a tracked score changes.
    """

    REFRESH_INTERVAL = 20
    GOAL_ANIMATION_SEC = 5.0
    SCORE_FLASH_SEC = 5.0
    SCORE_FLASH_HZ = 1

    def __init__(self, width: int, height: int, mode_manager: ModeManager) -> None:
        self.width = width
        self.height = height
        self.mode_manager = mode_manager
        self.allowed_chars = supported_characters(sizes=(5, 6), styles=("regular", "monospace"))
        self.last_refresh = 0.0
        self.last_payload: Any = None
        self.known_scores: dict[Any, tuple[Any, Any]] = {}
        self.goal_animation_until = 0.0
        self.score_flash_until = 0.0
        self.flashing_score_sides: dict[Any, tuple[bool, bool]] = {}
        self.frame = np.zeros((height, width), dtype=np.uint8)
        self._refresh_lock = threading.Lock()
        self._refresh_in_flight = False
        self._pending_payload = None

    def _sanitize(self, value):
        text = (value or "").upper()
        return "".join(ch for ch in text if ch in self.allowed_chars)

    def _short_status(self, match, selection):
        if match is None:
            return "NO DATA"

        status = (match.get("status") or "").upper()
        home_pen = match.get("home_penalty_score")
        away_pen = match.get("away_penalty_score")
        minute_raw = (match.get("minute") or "").strip()
        minute_match = re.search(r"(\d{1,3})(?:\s*\+\s*(\d{1,2}))?", minute_raw)
        minute_text = None
        minute_value = None
        if minute_match:
            minute_value = int(minute_match.group(1))
            added = minute_match.group(2)
            minute_text = f"{minute_value}+{int(added)}" if added else str(minute_value)

        has_penalty_score = home_pen is not None and away_pen is not None

        if status == "AET":
            return "AET"

        if has_penalty_score:
            return f"PEN {home_pen}:{away_pen}"

        if status == "PEN":
            return "PEN"

        if selection == "live":
            if status == "HT":
                return "HT"
            if status == "ET":
                if minute_text:
                    # During extra time stoppage, minute alone carries the phase.
                    if "+" in minute_text:
                        return minute_text
                    return f"{minute_text} ET"
                return "ET"
            if minute_text:
                return minute_text
            if status and status not in {"NS", "FT"}:
                return self._sanitize(status)[:7] or "--"
            return "--"

        if selection == "latest_finished":
            return "FT"

        if status:
            return self._sanitize(status)[:7]
        return "NO DATA"

    def _live_matches(self, payload):
        events = payload.get("events") or []
        seen_ids = set()
        live = []
        for event in events:
            if event.get("status_bucket") != "live":
                continue
            event_id = event.get("event_id")
            if event_id and event_id in seen_ids:
                continue
            if event_id:
                seen_ids.add(event_id)
            live.append(event)

        def kickoff_key(event):
            kickoff = event.get("kickoff_utc")
            if kickoff is None:
                return float("-inf")
            return kickoff.timestamp()

        live.sort(key=kickoff_key, reverse=True)
        return live[:2]

    def _team_code(self, match, field_name):
        code = self._sanitize(match.get(field_name) or "")[:3]
        if code:
            return code
        return "UNK"

    def _render_two_live_matches(self, frame, matches):
        for idx, match in enumerate(matches):
            # Bottom band is nudged down one row so there is a blank line of
            # space beneath the divider at row 13.
            band_top = idx * 14 + (1 if idx else 0)

            home = self._team_code(match, "home_code")
            away = self._team_code(match, "away_code")
            home_score = match.get("home_score")
            away_score = match.get("away_score")
            if home_score is None:
                home_score = 0
            if away_score is None:
                away_score = 0

            teams_text = self._sanitize(f"{home} {away}")
            event_id = match.get("event_id")

            write_centered(frame, teams_text, y=band_top, size=5, style="regular")
            self._render_score(frame, event_id, home_score, away_score, y=band_top + 6)

        # Divider between top and bottom match bands.
        frame[13, :] = 1

    def _render_match(self, frame, payload):
        live_matches = self._live_matches(payload)
        if len(live_matches) == 2:
            self._render_two_live_matches(frame, live_matches)
            return

        match = payload.get("selected")
        selection = payload.get("selection", "none")

        if match is None:
            write(frame, "NO DATA", x=2, y=11, size=5, style="regular")
            return

        home = self._team_code(match, "home_code")
        away = self._team_code(match, "away_code")
        home_score = match.get("home_score")
        away_score = match.get("away_score")

        if home_score is None:
            home_score = 0 if selection != "none" else "-"
        if away_score is None:
            away_score = 0 if selection != "none" else "-"

        teams_text = self._sanitize(f"{home} {away}")
        status_text = self._short_status(match, selection)
        event_id = match.get("event_id")

        self._render_score(frame, event_id, home_score, away_score, y=4)
        write_centered(frame, teams_text, y=14, size=5, style="regular")
        write_centered(frame, status_text, y=20, size=5, style="regular")

    def _score_visibility(self, event_id):
        now = time.time()
        if now < self.goal_animation_until:
            return True, True

        if now >= self.score_flash_until:
            return True, True

        flash_sides = self.flashing_score_sides.get(event_id)
        if not event_id or flash_sides is None:
            return True, True

        # Anchor the phase to time remaining so the flash always finishes on a
        # visible half-period; otherwise it can end with the score hidden.
        remaining = self.score_flash_until - now
        phase_on = int(remaining * self.SCORE_FLASH_HZ) % 2 == 0
        home_changed, away_changed = flash_sides
        show_home = phase_on if home_changed else True
        show_away = phase_on if away_changed else True
        return show_home, show_away

    def _render_score(self, frame, event_id, home_score, away_score, y):
        home_text = self._sanitize(str(home_score))
        away_text = self._sanitize(str(away_score))
        score_text = f"{home_text}:{away_text}"

        x = center_x(self.width, score_text, font="scoreline", size=6, style="regular")
        home_w = width(home_text, font="scoreline", size=6, style="regular")
        colon_w = width(":", font="scoreline", size=6, style="regular")

        show_home, show_away = self._score_visibility(event_id)
        if show_home:
            write(frame, home_text, x=x, y=y, font="scoreline", size=6, style="regular")

        colon_x = x + home_w + 1
        write(frame, ":", x=colon_x, y=y, font="scoreline", size=6, style="regular")

        away_x = colon_x + colon_w + 1
        if show_away:
            write(frame, away_text, x=away_x, y=y, font="scoreline", size=6, style="regular")

    def _collect_score_snapshot(self, payload):
        snapshot = {}
        events = payload.get("events") or []
        for event in events:
            event_id = event.get("event_id")
            if not event_id:
                continue

            home_score = event.get("home_score")
            away_score = event.get("away_score")
            if home_score is None or away_score is None:
                continue
            snapshot[event_id] = (home_score, away_score)

        selected = payload.get("selected")
        if selected is not None and selected.get("event_id"):
            home_score = selected.get("home_score")
            away_score = selected.get("away_score")
            if home_score is not None and away_score is not None:
                snapshot[selected.get("event_id")] = (home_score, away_score)

        return snapshot

    def _update_goal_animation(self, payload):
        snapshot = self._collect_score_snapshot(payload)
        if not snapshot:
            return

        # First seen scores seed baseline and must not trigger animation.
        if not self.known_scores:
            self.known_scores = dict(snapshot)
            return

        scored_event_sides = {}
        for event_id, score in snapshot.items():
            previous = self.known_scores.get(event_id)
            if previous is None:
                continue
            home_scored = score[0] > previous[0]
            away_scored = score[1] > previous[1]
            if home_scored or away_scored:
                scored_event_sides[event_id] = (home_scored, away_scored)

        self.known_scores.update(snapshot)

        if scored_event_sides:
            now = time.time()
            self.goal_animation_until = now + self.GOAL_ANIMATION_SEC
            self.score_flash_until = self.goal_animation_until + self.SCORE_FLASH_SEC
            self.flashing_score_sides = scored_event_sides

    def _apply_goal_animation(self, frame):
        now = time.time()
        if now >= self.goal_animation_until:
            return

        elapsed = max(0.0, self.GOAL_ANIMATION_SEC - (self.goal_animation_until - now))

        # Concentric shockwave rings radiating outward from the centre.
        center_y = (self.height - 1) / 2.0
        center_x_px = (self.width - 1) / 2.0
        rows = np.arange(self.height).reshape(-1, 1)
        cols = np.arange(self.width).reshape(1, -1)
        distance = np.sqrt((rows - center_y) ** 2 + (cols - center_x_px) ** 2)

        wave = np.sin(distance * 1.3 - elapsed * 12.0)
        frame[:] = (wave > 0.35).astype(np.uint8)

        # Steady banner keeps "GOAL" readable while the rings sweep behind it.
        frame[9:19, :] = 0
        frame[9, :] = 1
        frame[18, :] = 1
        write(
            frame,
            "GOAL",
            x=center_x(self.width, "GOAL", size=6, style="regular"),
            y=11,
            size=6,
            style="regular",
        )

    def _refresh_worker(self):
        payload = None
        try:
            payload = get_worldcup_scorecard()
        except Exception:
            logger.exception("WorldCup refresh failed")

        with self._refresh_lock:
            if payload is not None:
                self._pending_payload = payload
            self._refresh_in_flight = False

    def _drain_pending_payload(self):
        pending = None
        with self._refresh_lock:
            pending = self._pending_payload
            self._pending_payload = None

        if pending is None:
            return

        self._update_goal_animation(pending)
        # Keep last known good selection during transient API failures.
        if pending.get("selected") is not None or self.last_payload is None:
            self.last_payload = pending

    def _refresh_if_needed(self):
        self._drain_pending_payload()

        now = time.time()
        if self.last_payload is None or now - self.last_refresh >= self.REFRESH_INTERVAL:
            self.last_refresh = now
            should_refresh = False
            with self._refresh_lock:
                if not self._refresh_in_flight:
                    self._refresh_in_flight = True
                    should_refresh = True

            if should_refresh:
                try:
                    threading.Thread(target=self._refresh_worker, daemon=True).start()
                except Exception:
                    with self._refresh_lock:
                        self._refresh_in_flight = False

    def get_frame(self, pose_results: object) -> Frame:
        """Refresh the scorecard if due and render the selected match."""
        del pose_results  # Pose is not needed for this information mode.

        self._refresh_if_needed()
        self.frame = np.zeros((self.height, self.width), dtype=np.uint8)
        if self.last_payload is None:
            write(self.frame, "LOADING", x=2, y=11, size=5, style="regular")
            return self.frame

        self._render_match(self.frame, self.last_payload)
        self._apply_goal_animation(self.frame)
        return self.frame
