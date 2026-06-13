import logging
import re
from datetime import datetime, timezone

import requests


logger = logging.getLogger(__name__)

BASE_URL = "https://www.thesportsdb.com/api/v1/json/3"
LEAGUE_NAME = "FIFA World Cup"
REQUEST_TIMEOUT_SEC = 4

_cached_worldcup_league_id = None


def _safe_int(value):
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _event_datetime(event):
    date_str = event.get("dateEvent") or event.get("strDate")
    time_str = event.get("strTime") or "00:00:00"
    if not date_str:
        return None

    # TheSportsDB commonly uses HH:MM:SS in UTC.
    stamp = f"{date_str} {time_str[:8]}"
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(stamp, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _status_bucket(event):
    status = (event.get("strStatus") or "").upper()
    if status in {"FT", "AET", "FT_PEN", "PEN", "CANC", "ABD", "AWD", "WO"}:
        return "finished"
    if status in {"NS", "TBD", "PST"}:
        return "scheduled"
    if status:
        return "live"

    home = _safe_int(event.get("intHomeScore"))
    away = _safe_int(event.get("intAwayScore"))
    if home is not None and away is not None:
        return "finished"
    return "scheduled"


def _team_code(team_name):
    if not team_name:
        return "---"

    clean = re.sub(r"[^A-Z ]", "", team_name.upper()).strip()
    words = [w for w in clean.split() if w]
    if len(words) >= 2:
        initials = "".join(word[0] for word in words[:3])
        if len(initials) >= 2:
            return initials[:3]

    compact = "".join(ch for ch in clean if ch.isalpha())
    if not compact:
        return "---"
    return compact[:3]


def _normalize_event(event):
    home_team = (event.get("strHomeTeam") or "").upper()
    away_team = (event.get("strAwayTeam") or "").upper()
    home_score = _safe_int(event.get("intHomeScore"))
    away_score = _safe_int(event.get("intAwayScore"))
    home_penalty_score = _safe_int(
        event.get("intHomePenaltyScore")
        or event.get("intHomePenScore")
    )
    away_penalty_score = _safe_int(
        event.get("intAwayPenaltyScore")
        or event.get("intAwayPenScore")
    )
    status = (event.get("strStatus") or "NS").upper()
    minute = (event.get("strProgress") or "").strip()
    dt_utc = _event_datetime(event)

    return {
        "event_id": event.get("idEvent"),
        "league": event.get("strLeague"),
        "home_team": home_team,
        "away_team": away_team,
        "home_code": _team_code(home_team),
        "away_code": _team_code(away_team),
        "home_score": home_score,
        "away_score": away_score,
        "home_penalty_score": home_penalty_score,
        "away_penalty_score": away_penalty_score,
        "status": status,
        "status_bucket": _status_bucket(event),
        "minute": minute,
        "kickoff_utc": dt_utc,
    }


def _fetch_json(path, params=None):
    url = f"{BASE_URL}/{path}"
    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SEC)
    response.raise_for_status()
    return response.json()


def _get_worldcup_league_id():
    global _cached_worldcup_league_id
    if _cached_worldcup_league_id is not None:
        return _cached_worldcup_league_id

    payload = _fetch_json("search_all_teams.php", params={"l": LEAGUE_NAME})
    teams = payload.get("teams") or []
    for team in teams:
        league_name = (team.get("strLeague") or "").strip()
        if league_name == LEAGUE_NAME:
            _cached_worldcup_league_id = team.get("idLeague")
            break

    return _cached_worldcup_league_id


def _choose_latest_finished(events):
    finished = [e for e in events if e.get("status_bucket") == "finished"]
    if not finished:
        return None

    def key(event):
        dt = event.get("kickoff_utc")
        return dt or datetime.min.replace(tzinfo=timezone.utc)

    return max(finished, key=key)


def _choose_live(events):
    live = [e for e in events if e.get("status_bucket") == "live"]
    if not live:
        return None

    def key(event):
        # Prefer events that started most recently if multiple are live.
        dt = event.get("kickoff_utc")
        return dt or datetime.min.replace(tzinfo=timezone.utc)

    return max(live, key=key)


def get_worldcup_scorecard():
    """Fetch live FIFA World Cup score data.

    Returns a dict:
    {
        "selected": normalized_event | None,
        "selection": "live" | "latest_finished" | "none",
        "events": list[normalized_event],
        "error": str (optional)
    }
    """

    try:
        events = []

        league_id = _get_worldcup_league_id()
        if league_id:
            past_payload = _fetch_json("eventspastleague.php", params={"id": league_id})
            past_events = past_payload.get("events") or []
            events.extend(_normalize_event(event) for event in past_events)

            next_payload = _fetch_json("eventsnextleague.php", params={"id": league_id})
            next_events = next_payload.get("events") or []
            events.extend(_normalize_event(event) for event in next_events)
        else:
            logger.warning("World Cup league id was not found on TheSportsDB")

        if not events:
            return {
                "selected": None,
                "selection": "none",
                "events": [],
            }

        selected_live = _choose_live(events)
        if selected_live is not None:
            return {
                "selected": selected_live,
                "selection": "live",
                "events": events,
            }

        selected_finished = _choose_latest_finished(events)
        if selected_finished is not None:
            return {
                "selected": selected_finished,
                "selection": "latest_finished",
                "events": events,
            }

        return {
            "selected": None,
            "selection": "none",
            "events": events,
        }
    except requests.RequestException as exc:
        logger.warning("World Cup API request failed: %s", exc)
        return {
            "selected": None,
            "selection": "none",
            "events": [],
            "error": f"API request failed: {exc}",
        }
    except Exception as exc:  # pragma: no cover - defensive runtime safeguard
        logger.exception("Unexpected world cup service failure")
        return {
            "selected": None,
            "selection": "none",
            "events": [],
            "error": str(exc),
        }