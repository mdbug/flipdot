import time
import re

import numpy as np

from app.services.text import center_x, supported_characters, width, write, write_centered
from app.services.worldcup import get_worldcup_scorecard


class WorldCup:
    REFRESH_INTERVAL = 20
    GOAL_ANIMATION_SEC = 1.2
    GOAL_FLASH_HZ = 6
    SCORE_FLASH_SEC = 2.0
    SCORE_FLASH_HZ = 4

    def __init__(self, width, height, mode_manager):
        self.width = width
        self.height = height
        self.mode_manager = mode_manager
        self.allowed_chars = supported_characters(sizes=(5, 6), styles=("regular", "monospace"))
        self.last_refresh = 0.0
        self.last_payload = None
        self.known_scores = {}
        self.goal_animation_until = 0.0
        self.score_flash_until = 0.0
        self.flashing_score_sides = {}
        self.frame = np.zeros((height, width), dtype=np.uint8)

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
        blink_on = int(time.time() * 2) % 2 == 0

        for idx, match in enumerate(matches):
            band_top = idx * 14

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

            if blink_on:
                frame[band_top, self.width - 1] = 1

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

        phase_on = int(now * self.SCORE_FLASH_HZ) % 2 == 0
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
        if time.time() >= self.goal_animation_until:
            return

        phase_on = int(time.time() * self.GOAL_FLASH_HZ) % 2 == 0
        if phase_on:
            frame[0, :] = 1
            frame[-1, :] = 1
            frame[:, 0] = 1
            frame[:, -1] = 1
            frame[9:19, :] = 0
            write(
                frame,
                "GOAL",
                x=center_x(self.width, "GOAL", size=6, style="regular"),
                y=11,
                size=6,
                style="regular",
            )
            return

        checker = (np.indices(frame.shape).sum(axis=0) % 2).astype(np.uint8)
        frame ^= checker

    def _refresh_if_needed(self):
        now = time.time()
        if self.last_payload is None or now - self.last_refresh >= self.REFRESH_INTERVAL:
            payload = get_worldcup_scorecard()
            self.last_refresh = now
            self._update_goal_animation(payload)
            # Keep last known good selection during transient API failures.
            if payload.get("selected") is not None or self.last_payload is None:
                self.last_payload = payload

    def get_frame(self, pose_results):
        del pose_results  # Pose is not needed for this information mode.

        self._refresh_if_needed()
        self.frame = np.zeros((self.height, self.width), dtype=np.uint8)
        if self.last_payload is None:
            write(self.frame, "LOADING", x=2, y=11, size=5, style="regular")
            return self.frame

        self._render_match(self.frame, self.last_payload)
        self._apply_goal_animation(self.frame)
        return self.frame