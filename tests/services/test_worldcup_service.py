from datetime import datetime, timedelta, timezone
import app.services.worldcup as worldcup_module


def test_safe_int_handles_none_empty_and_invalid():
    assert worldcup_module._safe_int(None) is None
    assert worldcup_module._safe_int("") is None
    assert worldcup_module._safe_int("12") == 12
    assert worldcup_module._safe_int("x") is None


def test_event_datetime_parses_utc_timestamp():
    event = {"fixture": {"date": "2026-06-13T15:30:00+00:00"}}
    dt = worldcup_module._event_datetime(event)
    assert dt == datetime(2026, 6, 13, 15, 30, 0, tzinfo=timezone.utc)


def test_status_bucket_covers_live_finished_and_scheduled():
    assert worldcup_module._status_bucket({"fixture": {"status": {"short": "FT"}}}) == "finished"
    assert worldcup_module._status_bucket({"fixture": {"status": {"short": "NS"}}}) == "scheduled"
    assert worldcup_module._status_bucket({"fixture": {"status": {"short": "2H"}}}) == "live"
    assert worldcup_module._status_bucket({"goals": {"home": 1, "away": 0}}) == "finished"


def test_team_code_generation():
    assert worldcup_module._team_code("Brazil") == "BRA"
    assert worldcup_module._team_code("New Zealand") == "NZ"
    assert worldcup_module._team_code("") == "---"


def test_normalize_event_maps_fields_and_helpers():
    raw = {
        "fixture": {
            "id": "abc",
            "date": "2026-06-13T20:00:00+00:00",
            "status": {"short": "PEN", "elapsed": 120, "extra": 2},
        },
        "league": {"name": "World Cup"},
        "teams": {
            "home": {"name": "Brazil"},
            "away": {"name": "Argentina"},
        },
        "goals": {"home": 2, "away": 1},
        "score": {"penalty": {"home": 4, "away": 3}},
    }
    event = worldcup_module._normalize_event(raw)

    assert event["event_id"] == "abc"
    assert event["home_team"] == "BRAZIL"
    assert event["away_team"] == "ARGENTINA"
    assert event["home_score"] == 2
    assert event["away_score"] == 1
    assert event["home_penalty_score"] == 4
    assert event["away_penalty_score"] == 3
    assert event["status_bucket"] == "finished"
    assert event["minute"] == "120+2"
    assert event["home_code"] == "BRA"
    assert event["away_code"] == "ARG"
    assert event["kickoff_utc"] == datetime(2026, 6, 13, 20, 0, 0, tzinfo=timezone.utc)


def test_get_worldcup_scorecard_prefers_live(monkeypatch):
    monkeypatch.setattr(
        worldcup_module,
        "_get_worldcup_context",
        lambda: {"league_id": 1, "seasons": [2026, 2022]},
    )

    def fake_fetch(path, params=None):
        if path != "fixtures":
            raise AssertionError(f"unexpected path {path}")
        if params == {"league": 1, "live": "all"}:
            return {
                "response": [
                    {
                        "fixture": {
                            "id": 501,
                            "date": "2026-06-13T20:00:00+00:00",
                            "status": {"short": "2H", "elapsed": 77, "extra": None},
                        },
                        "league": {"name": "World Cup"},
                        "teams": {
                            "home": {"name": "Brazil"},
                            "away": {"name": "Argentina"},
                        },
                        "goals": {"home": 1, "away": 1},
                        "score": {"penalty": {"home": None, "away": None}},
                    }
                ]
            }
        if params == {"league": 1, "season": 2026}:
            return {"response": []}
        if params == {"league": 1, "season": 2026, "last": 20}:
            return {"response": []}
        raise AssertionError(f"unexpected params {params}")

    monkeypatch.setattr(worldcup_module, "_fetch_json", fake_fetch)
    worldcup_module._schedule_windows_utc = []
    worldcup_module._next_schedule_refresh_after_mono = 0.0
    worldcup_module._cached_scorecard = None
    worldcup_module._next_fetch_after_mono = 0.0
    scorecard = worldcup_module.get_worldcup_scorecard()
    assert scorecard["selection"] == "live"
    assert scorecard["selected"]["event_id"] == 501
    assert len(scorecard["events"]) == 1


def test_adaptive_interval_respects_daily_remaining(monkeypatch):
    monkeypatch.setitem(worldcup_module._rate_limit_state, "daily_remaining", 10)
    monkeypatch.setitem(worldcup_module._rate_limit_state, "minute_remaining", None)
    monkeypatch.setattr(worldcup_module, "_seconds_until_utc_day_end", lambda now=None: 10000.0)

    interval = worldcup_module._adaptive_interval_sec("live")
    # With active/idle split and no active windows, idle budget becomes dominant.
    # 5 usable requests -> idle alloc 2, so 10000/2=5000 and then capped at 3600.
    assert interval == 3600.0


def test_update_rate_limit_state_parses_expected_headers():
    worldcup_module._update_rate_limit_state(
        {
            "x-ratelimit-requests-limit": "100",
            "x-ratelimit-requests-remaining": "97",
            "x-ratelimit-limit": "10",
            "x-ratelimit-remaining": "8",
        }
    )

    assert worldcup_module._rate_limit_state["daily_limit"] == 100
    assert worldcup_module._rate_limit_state["daily_remaining"] == 97
    assert worldcup_module._rate_limit_state["minute_limit"] == 10
    assert worldcup_module._rate_limit_state["minute_remaining"] == 8


def test_get_worldcup_scorecard_uses_cache(monkeypatch):
    calls = {"count": 0}

    def fake_fetch(path, params=None):
        calls["count"] += 1
        if path == "leagues":
            return {
                "response": [
                    {
                        "league": {"id": 1, "name": "World Cup"},
                        "country": {"name": "World"},
                        "seasons": [{"year": 2026}],
                    }
                ]
            }
        if path == "fixtures":
            return {"response": []}
        raise AssertionError(path)

    now = {"value": 100.0}
    monkeypatch.setattr(worldcup_module.time, "monotonic", lambda: now["value"])
    monkeypatch.setattr(worldcup_module, "_fetch_json", fake_fetch)

    worldcup_module._cached_worldcup_context = None
    worldcup_module._cached_scorecard = None
    worldcup_module._next_fetch_after_mono = 0.0
    worldcup_module._schedule_windows_utc = []
    worldcup_module._next_schedule_refresh_after_mono = 0.0
    worldcup_module._rate_limit_state["daily_remaining"] = None
    worldcup_module._rate_limit_state["minute_remaining"] = None

    first = worldcup_module.get_worldcup_scorecard()
    assert first["selection"] == "none"
    first_call_count = calls["count"]
    assert first_call_count >= 2

    now["value"] = 120.0
    second = worldcup_module.get_worldcup_scorecard()
    assert second["selection"] == "none"
    assert calls["count"] == first_call_count


def test_choose_live_and_latest_finished():
    e1 = {
        "event_id": "1",
        "status_bucket": "live",
        "kickoff_utc": datetime(2026, 6, 13, 18, 0, tzinfo=timezone.utc),
    }
    e2 = {
        "event_id": "2",
        "status_bucket": "live",
        "kickoff_utc": datetime(2026, 6, 13, 19, 0, tzinfo=timezone.utc),
    }
    f1 = {
        "event_id": "3",
        "status_bucket": "finished",
        "kickoff_utc": datetime(2026, 6, 12, 19, 0, tzinfo=timezone.utc),
    }
    f2 = {
        "event_id": "4",
        "status_bucket": "finished",
        "kickoff_utc": datetime(2026, 6, 13, 17, 0, tzinfo=timezone.utc),
    }

    assert worldcup_module._choose_live([e1, e2, f1])["event_id"] == "2"
    assert worldcup_module._choose_latest_finished([e1, f1, f2])["event_id"] == "4"
