from types import SimpleNamespace
import numpy as np

import app.modes.worldcup as worldcup_mode_module


def _payload(selected=None, events=None, selection="none"):
    return {
        "selected": selected,
        "selection": selection,
        "events": events or [],
    }


def test_refresh_if_needed_respects_refresh_interval(monkeypatch):
    now = {"value": 0.0}

    def fake_time():
        return now["value"]

    calls = {"count": 0}

    def fake_scorecard():
        calls["count"] += 1
        return _payload(selected={"event_id": "x", "home_score": 0, "away_score": 0})

    monkeypatch.setattr(worldcup_mode_module.time, "time", fake_time)
    monkeypatch.setattr(worldcup_mode_module, "get_worldcup_scorecard", fake_scorecard)

    mode = worldcup_mode_module.WorldCup(28, 28, SimpleNamespace())
    mode._refresh_if_needed()
    assert calls["count"] == 1

    now["value"] = 5.0
    mode._refresh_if_needed()
    assert calls["count"] == 1

    now["value"] = 31.0
    mode._refresh_if_needed()
    assert calls["count"] == 2


def test_short_status_for_penalties_and_extra_time():
    mode = worldcup_mode_module.WorldCup(28, 28, SimpleNamespace())

    pen_match = {"status": "PEN", "home_penalty_score": 4, "away_penalty_score": 3}
    assert mode._short_status(pen_match, "live") == "PEN 4:3"

    et_match = {"status": "ET", "minute": "105+1"}
    assert mode._short_status(et_match, "live") == "105+1"

    finished_match = {"status": "FT"}
    assert mode._short_status(finished_match, "latest_finished") == "FT"


def test_update_goal_animation_only_triggers_on_score_increase(monkeypatch):
    now = {"value": 100.0}

    def fake_time():
        return now["value"]

    monkeypatch.setattr(worldcup_mode_module.time, "time", fake_time)
    mode = worldcup_mode_module.WorldCup(28, 28, SimpleNamespace())

    baseline = _payload(
        selected={"event_id": "m1", "home_score": 1, "away_score": 0},
        events=[{"event_id": "m1", "home_score": 1, "away_score": 0}],
    )
    mode._update_goal_animation(baseline)
    assert mode.goal_animation_until == 0.0

    scored = _payload(
        selected={"event_id": "m1", "home_score": 2, "away_score": 0},
        events=[{"event_id": "m1", "home_score": 2, "away_score": 0}],
    )
    mode._update_goal_animation(scored)

    assert mode.goal_animation_until == 100.0 + mode.GOAL_ANIMATION_SEC
    assert mode.score_flash_until == mode.goal_animation_until + mode.SCORE_FLASH_SEC
    assert mode.flashing_score_sides == {"m1": (True, False)}


def test_score_visibility_flashes_only_changed_side(monkeypatch):
    now = {"value": 10.375}

    def fake_time():
        return now["value"]

    monkeypatch.setattr(worldcup_mode_module.time, "time", fake_time)
    mode = worldcup_mode_module.WorldCup(28, 28, SimpleNamespace())
    mode.goal_animation_until = 0.0
    mode.score_flash_until = 20.0
    mode.flashing_score_sides = {"m1": (True, False)}

    show_home, show_away = mode._score_visibility("m1")
    assert show_home is False
    assert show_away is True


def test_live_matches_returns_most_recent_two(monkeypatch):
    mode = worldcup_mode_module.WorldCup(28, 28, SimpleNamespace())
    payload = {
        "events": [
            {"event_id": "a", "status_bucket": "live", "kickoff_utc": SimpleNamespace(timestamp=lambda: 1.0)},
            {"event_id": "b", "status_bucket": "live", "kickoff_utc": SimpleNamespace(timestamp=lambda: 3.0)},
            {"event_id": "c", "status_bucket": "live", "kickoff_utc": SimpleNamespace(timestamp=lambda: 2.0)},
        ]
    }

    out = mode._live_matches(payload)
    assert [m["event_id"] for m in out] == ["b", "c"]


def test_render_match_no_data_path(monkeypatch):
    mode = worldcup_mode_module.WorldCup(28, 28, SimpleNamespace())
    frame = np.zeros((28, 28), dtype=np.uint8)
    mode._render_match(frame, {"selected": None, "selection": "none", "events": []})
    assert frame.sum() > 0


def test_apply_goal_animation_on_phase_draws_goal_frame(monkeypatch):
    now = {"value": 10.0}
    monkeypatch.setattr(worldcup_mode_module.time, "time", lambda: now["value"])
    mode = worldcup_mode_module.WorldCup(28, 28, SimpleNamespace())
    mode.goal_animation_until = 11.0
    frame = np.zeros((28, 28), dtype=np.uint8)

    mode._apply_goal_animation(frame)
    assert frame[0, :].sum() > 0
    assert frame[-1, :].sum() > 0
    assert frame[:, 0].sum() > 0
    assert frame[:, -1].sum() > 0
    assert frame.sum() > 0


def test_apply_goal_animation_checkerboard_branch(monkeypatch):
    now = {"value": 10.2}
    monkeypatch.setattr(worldcup_mode_module.time, "time", lambda: now["value"])
    mode = worldcup_mode_module.WorldCup(28, 28, SimpleNamespace())
    mode.goal_animation_until = 11.0
    frame = np.zeros((28, 28), dtype=np.uint8)

    mode._apply_goal_animation(frame)
    assert frame.sum() > 0
