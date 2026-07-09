import time
from datetime import datetime, timezone

import pytest

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
    monkeypatch.setattr(worldcup_module, "_get_espn_scorecard", lambda: None)
    monkeypatch.setattr(worldcup_module, "_get_fifa_scorecard", lambda: None)
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
    monkeypatch.setattr(worldcup_module, "_get_espn_scorecard", lambda: None)
    monkeypatch.setattr(worldcup_module, "_get_fifa_scorecard", lambda: None)
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


def test_get_worldcup_scorecard_prefers_fifa_source(monkeypatch):
    fifa_payload = {
        "selected": {"event_id": "fifa-1"},
        "selection": "live",
        "events": [{"event_id": "fifa-1", "status_bucket": "live"}],
        "rate_limit": {},
    }
    monkeypatch.setattr(worldcup_module, "_get_espn_scorecard", lambda: None)
    monkeypatch.setattr(worldcup_module, "_get_fifa_scorecard", lambda: fifa_payload)
    monkeypatch.setattr(
        worldcup_module,
        "_get_api_football_scorecard",
        lambda: {"selected": None, "selection": "none", "events": []},
    )

    scorecard = worldcup_module.get_worldcup_scorecard()
    assert scorecard is fifa_payload


def test_get_worldcup_scorecard_prefers_api_live_when_fifa_not_live(monkeypatch):
    fifa_payload = {
        "selected": {"event_id": "fifa-finished"},
        "selection": "latest_finished",
        "events": [{"event_id": "fifa-finished", "status_bucket": "finished"}],
        "rate_limit": {},
    }
    api_payload = {
        "selected": {"event_id": "api-live"},
        "selection": "live",
        "events": [{"event_id": "api-live", "status_bucket": "live"}],
        "rate_limit": {},
    }

    monkeypatch.setattr(worldcup_module, "_get_espn_scorecard", lambda: None)
    monkeypatch.setattr(worldcup_module, "_get_fifa_scorecard", lambda: fifa_payload)
    monkeypatch.setattr(worldcup_module, "_get_api_football_scorecard", lambda: api_payload)

    scorecard = worldcup_module.get_worldcup_scorecard()
    assert scorecard is api_payload


def test_get_worldcup_scorecard_falls_back_to_fifa_when_api_not_live(monkeypatch):
    fifa_payload = {
        "selected": {"event_id": "fifa-finished"},
        "selection": "latest_finished",
        "events": [{"event_id": "fifa-finished", "status_bucket": "finished"}],
        "rate_limit": {},
    }
    api_payload = {
        "selected": None,
        "selection": "none",
        "events": [],
        "rate_limit": {},
    }

    monkeypatch.setattr(worldcup_module, "_get_espn_scorecard", lambda: None)
    monkeypatch.setattr(worldcup_module, "_get_fifa_scorecard", lambda: fifa_payload)
    monkeypatch.setattr(worldcup_module, "_get_api_football_scorecard", lambda: api_payload)

    scorecard = worldcup_module.get_worldcup_scorecard()
    assert scorecard is fifa_payload


def test_normalize_fifa_event_maps_expected_fields():
    raw = {
        "IdMatch": "400021447",
        "CompetitionName": [{"Locale": "en-GB", "Description": "FIFA World Cup"}],
        "Date": "2026-06-13T19:00:00Z",
        "MatchTime": "45+3'",
        "MatchStatus": 3,
        "Period": 4,
        "HomeTeamPenaltyScore": None,
        "AwayTeamPenaltyScore": None,
        "HomeTeam": {
            "TeamName": [{"Locale": "en-GB", "Description": "Qatar"}],
            "Abbreviation": "QAT",
            "Score": 0,
        },
        "AwayTeam": {
            "TeamName": [{"Locale": "en-GB", "Description": "Switzerland"}],
            "Abbreviation": "SUI",
            "Score": 1,
        },
    }

    event = worldcup_module._normalize_fifa_event(raw)
    assert event["event_id"] == "400021447"
    assert event["league"] == "FIFA World Cup"
    assert event["home_team"] == "QATAR"
    assert event["away_team"] == "SWITZERLAND"
    assert event["home_code"] == "QAT"
    assert event["away_code"] == "SUI"
    assert event["home_score"] == 0
    assert event["away_score"] == 1
    assert event["status"] == "HT"
    assert event["status_bucket"] == "live"
    assert event["minute"] == "45+3"


def test_normalize_fifa_event_ignores_stale_halftime_period():
    raw = {
        "IdMatch": "400021447",
        "CompetitionName": [{"Locale": "en-GB", "Description": "FIFA World Cup"}],
        "Date": "2026-06-13T19:00:00Z",
        "MatchTime": "61'",
        "MatchStatus": 3,
        "Period": 5,
        "HomeTeam": {
            "TeamName": [{"Locale": "en-GB", "Description": "Qatar"}],
            "Abbreviation": "QAT",
            "Score": 0,
        },
        "AwayTeam": {
            "TeamName": [{"Locale": "en-GB", "Description": "Switzerland"}],
            "Abbreviation": "SUI",
            "Score": 1,
        },
    }

    event = worldcup_module._normalize_fifa_event(raw)
    assert event["status"] == "LIVE"
    assert event["status_bucket"] == "live"
    assert event["minute"] == "61"


def test_normalize_fifa_event_period_zero_maps_to_scheduled():
    raw = {
        "IdMatch": "400021448",
        "CompetitionName": [{"Locale": "en-GB", "Description": "FIFA World Cup"}],
        "Date": "2026-06-14T19:00:00Z",
        "MatchTime": "",
        "MatchStatus": 2,
        "Period": 0,
        "HomeTeam": {
            "TeamName": [{"Locale": "en-GB", "Description": "Team A"}],
            "Abbreviation": "AAA",
            "Score": None,
        },
        "AwayTeam": {
            "TeamName": [{"Locale": "en-GB", "Description": "Team B"}],
            "Abbreviation": "BBB",
            "Score": None,
        },
    }

    event = worldcup_module._normalize_fifa_event(raw)
    assert event["status"] == "NS"
    assert event["status_bucket"] == "scheduled"


def test_normalize_fifa_event_parses_stoppage_time_with_apostrophe_plus():
    raw = {
        "IdMatch": "400021447",
        "CompetitionName": [{"Locale": "en-GB", "Description": "FIFA World Cup"}],
        "Date": "2026-06-13T19:00:00Z",
        "MatchTime": "90'+6'",
        "MatchStatus": 3,
        "Period": 7,
        "HomeTeam": {
            "TeamName": [{"Locale": "en-GB", "Description": "Qatar"}],
            "Abbreviation": "QAT",
            "Score": 0,
        },
        "AwayTeam": {
            "TeamName": [{"Locale": "en-GB", "Description": "Switzerland"}],
            "Abbreviation": "SUI",
            "Score": 1,
        },
    }

    event = worldcup_module._normalize_fifa_event(raw)
    assert event["minute"] == "90+6"


def test_effective_fifa_match_ids_discovers_recent_upcoming_matches(monkeypatch):
    now_utc = datetime(2026, 6, 13, 20, 0, 0, tzinfo=timezone.utc)
    now_mono = 123.0

    worldcup_module._last_fifa_match_id_hints = ["400021447"]

    def fake_fetch(path, params=None):
        if not path.startswith("calendar/"):
            raise AssertionError(path)
        mid = int(path.split("/")[-1])
        if mid == 400021447:
            return {
                "IdMatch": "400021447",
                "IdCompetition": "17",
                "Date": "2026-06-13T19:00:00Z",
                "Period": 10,
                "MatchStatus": 0,
                "CompetitionName": [{"Locale": "en-GB", "Description": "FIFA World Cup"}],
            }
        if mid == 400021456:
            return {
                "IdMatch": "400021456",
                "IdCompetition": "17",
                "Date": "2026-06-13T22:00:00Z",
                "Period": 0,
                "MatchStatus": 1,
                "CompetitionName": [{"Locale": "en-GB", "Description": "FIFA World Cup"}],
            }
        return {"IdMatch": str(mid), "IdCompetition": "520", "Date": "2026-06-10T12:00:00Z"}

    monkeypatch.setattr(worldcup_module, "_fetch_fifa_json", fake_fetch)

    worldcup_module._discovered_fifa_match_ids = None
    worldcup_module._next_fifa_match_discovery_after_mono = 0.0

    out = worldcup_module._effective_fifa_match_ids(now_mono, now_utc)
    assert out[0] == "400021447"
    assert "400021456" in out


def test_effective_fifa_match_ids_empty_without_hints(monkeypatch):
    now_utc = datetime(2026, 6, 13, 20, 0, 0, tzinfo=timezone.utc)
    now_mono = 123.0

    worldcup_module._last_fifa_match_id_hints = []
    worldcup_module._discovered_fifa_match_ids = None
    worldcup_module._next_fifa_match_discovery_after_mono = 0.0

    called = {"value": False}

    def fake_fetch(path, params=None):
        called["value"] = True
        return {}

    monkeypatch.setattr(worldcup_module, "_fetch_fifa_json", fake_fetch)

    out = worldcup_module._effective_fifa_match_ids(now_mono, now_utc)
    assert out == []
    assert called["value"] is False


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


def _espn_event(
    event_id,
    state,
    home,
    away,
    *,
    home_score=None,
    away_score=None,
    status_name="STATUS_IN_PROGRESS",
    display_clock="0'",
    date="2026-06-26T19:00Z",
    home_shootout=None,
    away_shootout=None,
):
    return {
        "id": event_id,
        "name": f"{home} vs {away}",
        "date": date,
        "competitions": [
            {
                "status": {
                    "displayClock": display_clock,
                    "type": {"state": state, "name": status_name},
                },
                "competitors": [
                    {
                        "homeAway": "home",
                        "score": home_score,
                        "shootoutScore": home_shootout,
                        "team": {"displayName": home[0], "abbreviation": home[1]},
                    },
                    {
                        "homeAway": "away",
                        "score": away_score,
                        "shootoutScore": away_shootout,
                        "team": {"displayName": away[0], "abbreviation": away[1]},
                    },
                ],
            }
        ],
    }


def test_normalize_espn_event_live_maps_fields():
    raw = _espn_event(
        "401",
        "in",
        ("France", "FRA"),
        ("Norway", "NOR"),
        home_score=2,
        away_score=1,
        status_name="STATUS_IN_PROGRESS",
        display_clock="67'",
    )
    event = worldcup_module._normalize_espn_event(raw)

    assert event["event_id"] == "401"
    assert event["home_code"] == "FRA"
    assert event["away_code"] == "NOR"
    assert event["home_team"] == "FRANCE"
    assert event["home_score"] == 2
    assert event["away_score"] == 1
    assert event["status"] == "LIVE"
    assert event["status_bucket"] == "live"
    assert event["minute"] == "67"
    assert event["kickoff_utc"] == datetime(2026, 6, 26, 19, 0, tzinfo=timezone.utc)


def test_normalize_espn_event_halftime_and_scheduled():
    ht = worldcup_module._normalize_espn_event(
        _espn_event("1", "in", ("A", "AAA"), ("B", "BBB"), status_name="STATUS_HALFTIME")
    )
    assert ht["status"] == "HT"
    assert ht["status_bucket"] == "live"

    scheduled = worldcup_module._normalize_espn_event(
        _espn_event("2", "pre", ("A", "AAA"), ("B", "BBB"), display_clock="0'")
    )
    assert scheduled["status"] == "NS"
    assert scheduled["status_bucket"] == "scheduled"
    assert scheduled["minute"] == ""


def test_normalize_espn_event_finished_with_penalty_shootout():
    raw = _espn_event(
        "3",
        "post",
        ("Spain", "ESP"),
        ("Italy", "ITA"),
        home_score=1,
        away_score=1,
        status_name="STATUS_FINAL_PENALTIES",
        home_shootout=4,
        away_shootout=3,
    )
    event = worldcup_module._normalize_espn_event(raw)

    assert event["status_bucket"] == "finished"
    assert event["status"] == "FT"
    assert event["home_penalty_score"] == 4
    assert event["away_penalty_score"] == 3


def test_normalize_espn_event_falls_back_to_team_code_without_abbreviation():
    raw = _espn_event("4", "in", ("United States", ""), ("South Korea", ""))
    event = worldcup_module._normalize_espn_event(raw)
    assert event["home_code"] == "US"
    assert event["away_code"] == "SK"


def test_get_espn_scorecard_selects_live(monkeypatch):
    payload = {
        "events": [
            _espn_event(
                "10",
                "post",
                ("A", "AAA"),
                ("B", "BBB"),
                home_score=0,
                away_score=0,
                status_name="STATUS_FULL_TIME",
                date="2026-06-25T19:00Z",
            ),
            _espn_event(
                "11",
                "in",
                ("C", "CCC"),
                ("D", "DDD"),
                home_score=1,
                away_score=0,
                display_clock="55'",
                date="2026-06-26T19:00Z",
            ),
        ]
    }
    monkeypatch.setattr(worldcup_module, "_fetch_espn_json", lambda url: payload)
    worldcup_module._cached_espn_scorecard = None
    worldcup_module._next_espn_fetch_after_mono = 0.0

    card = worldcup_module._get_espn_scorecard()
    assert card["selection"] == "live"
    assert card["selected"]["event_id"] == "11"
    assert len(card["events"]) == 2


def test_get_worldcup_scorecard_prefers_espn_live(monkeypatch):
    espn_payload = {
        "selected": {"event_id": "espn-live"},
        "selection": "live",
        "events": [{"event_id": "espn-live", "status_bucket": "live"}],
    }
    monkeypatch.setattr(worldcup_module, "_get_espn_scorecard", lambda: espn_payload)

    def fail(*args, **kwargs):
        raise AssertionError("fallback sources must not be called when ESPN is live")

    monkeypatch.setattr(worldcup_module, "_get_fifa_scorecard", fail)
    monkeypatch.setattr(worldcup_module, "_get_api_football_scorecard", fail)

    assert worldcup_module.get_worldcup_scorecard() is espn_payload


def test_get_worldcup_scorecard_falls_back_when_espn_raises(monkeypatch):
    fifa_payload = {
        "selected": {"event_id": "fifa-live"},
        "selection": "live",
        "events": [{"event_id": "fifa-live", "status_bucket": "live"}],
        "rate_limit": {},
    }

    def boom():
        raise worldcup_module.requests.RequestException("ESPN down")

    monkeypatch.setattr(worldcup_module, "_get_espn_scorecard", boom)
    monkeypatch.setattr(worldcup_module, "_get_fifa_scorecard", lambda: fifa_payload)
    monkeypatch.setattr(
        worldcup_module,
        "_get_api_football_scorecard",
        lambda: {"selected": None, "selection": "none", "events": []},
    )

    assert worldcup_module.get_worldcup_scorecard() is fifa_payload


def test_get_espn_scorecard_backs_off_after_failure(monkeypatch):
    calls = {"n": 0}

    def boom(url):
        calls["n"] += 1
        raise worldcup_module.requests.RequestException("ESPN down")

    monkeypatch.setattr(worldcup_module, "_fetch_espn_json", boom)
    monkeypatch.setattr(worldcup_module, "_cached_espn_scorecard", None)
    monkeypatch.setattr(worldcup_module, "_next_espn_fetch_after_mono", 0.0)

    with pytest.raises(worldcup_module.requests.RequestException):
        worldcup_module._get_espn_scorecard()
    assert calls["n"] == 1
    # The failure must arm the retry interval even though nothing was cached.
    assert worldcup_module._next_espn_fetch_after_mono > time.monotonic()

    # Within the backoff window the endpoint is not re-tried (no second
    # timeout paid); the absent cache is served as None instead.
    assert worldcup_module._get_espn_scorecard() is None
    assert calls["n"] == 1


def test_get_fifa_scorecard_backs_off_after_failure(monkeypatch):
    calls = {"n": 0}

    def boom(path, params=None):
        calls["n"] += 1
        raise worldcup_module.requests.RequestException("FIFA down")

    monkeypatch.setattr(worldcup_module, "_fetch_fifa_json", boom)
    monkeypatch.setattr(worldcup_module, "_cached_fifa_scorecard", None)
    monkeypatch.setattr(worldcup_module, "_next_fifa_fetch_after_mono", 0.0)
    # Pre-discovered match ids keep the (already-throttled) discovery pass out
    # of this test; only the live-fetch loop runs.
    monkeypatch.setattr(worldcup_module, "_discovered_fifa_match_ids", ["12345"])
    monkeypatch.setattr(
        worldcup_module, "_next_fifa_match_discovery_after_mono", time.monotonic() + 3600
    )

    with pytest.raises(worldcup_module.requests.RequestException):
        worldcup_module._get_fifa_scorecard()
    assert calls["n"] == 1
    assert worldcup_module._next_fifa_fetch_after_mono > time.monotonic()

    assert worldcup_module._get_fifa_scorecard() is None
    assert calls["n"] == 1
