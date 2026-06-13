from datetime import datetime, timezone

import app.services.worldcup as worldcup_module


def test_safe_int_handles_none_empty_and_invalid():
    assert worldcup_module._safe_int(None) is None
    assert worldcup_module._safe_int("") is None
    assert worldcup_module._safe_int("12") == 12
    assert worldcup_module._safe_int("x") is None


def test_event_datetime_parses_utc_timestamp():
    event = {"dateEvent": "2026-06-13", "strTime": "15:30:00"}
    dt = worldcup_module._event_datetime(event)
    assert dt == datetime(2026, 6, 13, 15, 30, 0, tzinfo=timezone.utc)


def test_status_bucket_covers_live_finished_and_scheduled():
    assert worldcup_module._status_bucket({"strStatus": "FT"}) == "finished"
    assert worldcup_module._status_bucket({"strStatus": "NS"}) == "scheduled"
    assert worldcup_module._status_bucket({"strStatus": "2H"}) == "live"
    assert worldcup_module._status_bucket({"intHomeScore": "1", "intAwayScore": "0"}) == "finished"


def test_team_code_generation():
    assert worldcup_module._team_code("Brazil") == "BRA"
    assert worldcup_module._team_code("New Zealand") == "NZ"
    assert worldcup_module._team_code("") == "---"


def test_normalize_event_maps_fields_and_helpers():
    raw = {
        "idEvent": "abc",
        "strLeague": "FIFA World Cup",
        "strHomeTeam": "Brazil",
        "strAwayTeam": "Argentina",
        "intHomeScore": "2",
        "intAwayScore": "1",
        "intHomePenaltyScore": "4",
        "intAwayPenaltyScore": "3",
        "strStatus": "PEN",
        "strProgress": "90+2",
        "dateEvent": "2026-06-13",
        "strTime": "20:00:00",
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
    assert event["home_code"] == "BRA"
    assert event["away_code"] == "ARG"
    assert event["kickoff_utc"] == datetime(2026, 6, 13, 20, 0, 0, tzinfo=timezone.utc)


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
