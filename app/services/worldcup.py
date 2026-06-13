import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

import requests


logger = logging.getLogger(__name__)

BASE_URL = os.getenv("API_FOOTBALL_BASE_URL", "https://v3.football.api-sports.io").rstrip("/")
LEAGUE_NAME = "World Cup"
REQUEST_TIMEOUT_SEC = 4

# Keep a safety reserve so request spikes do not exhaust the daily quota.
DAILY_REQUEST_RESERVE = 5
MIN_RETRY_SEC = 60
MAX_RETRY_SEC = 60 * 60

BASE_INTERVAL_LIVE_SEC = 60
BASE_INTERVAL_FINISHED_SEC = 12 * 60
BASE_INTERVAL_NONE_SEC = 15 * 60
FALLBACK_LOOKUP_COOLDOWN_SEC = 6 * 60 * 60
STATUS_PROBE_COOLDOWN_SEC = 6 * 60 * 60
SCHEDULE_REFRESH_SEC = 24 * 60 * 60
MATCH_PRE_WINDOW_SEC = 15 * 60
MATCH_POST_WINDOW_SEC = 2 * 60 * 60
ACTIVE_BUDGET_SHARE = 0.75

_cached_worldcup_context = None
_cached_scorecard = None
_next_fetch_after_mono = 0.0
_next_fallback_lookup_after_mono = 0.0
_next_status_probe_after_mono = 0.0
_next_schedule_refresh_after_mono = 0.0
_schedule_windows_utc = []
_rate_limit_state = {
    "daily_limit": None,
    "daily_remaining": None,
    "minute_limit": None,
    "minute_remaining": None,
    "updated_at": None,
}


def _safe_int(value):
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _header_int(headers, *names):
    for name in names:
        value = headers.get(name)
        if value is None:
            continue
        parsed = _safe_int(value)
        if parsed is not None:
            return parsed
    return None


def _update_rate_limit_state(headers):
    daily_limit = _header_int(headers, "x-ratelimit-requests-limit")
    daily_remaining = _header_int(headers, "x-ratelimit-requests-remaining")
    minute_limit = _header_int(headers, "X-RateLimit-Limit", "x-ratelimit-limit")
    minute_remaining = _header_int(headers, "X-RateLimit-Remaining", "x-ratelimit-remaining")

    _rate_limit_state["daily_limit"] = daily_limit
    _rate_limit_state["daily_remaining"] = daily_remaining
    _rate_limit_state["minute_limit"] = minute_limit
    _rate_limit_state["minute_remaining"] = minute_remaining
    _rate_limit_state["updated_at"] = datetime.now(timezone.utc)


def _seconds_until_utc_day_end(now=None):
    if now is None:
        now = datetime.now(timezone.utc)
    midnight_tomorrow = datetime(now.year, now.month, now.day, tzinfo=timezone.utc) + timedelta(days=1)
    return max(1.0, (midnight_tomorrow - now).total_seconds())


def _merge_windows(windows):
    if not windows:
        return []

    ordered = sorted(windows, key=lambda w: w[0])
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _schedule_windows_from_fixtures(fixtures):
    windows = []
    for fixture in fixtures:
        kickoff = _event_datetime(fixture)
        if kickoff is None:
            continue
        windows.append(
            (
                kickoff - timedelta(seconds=MATCH_PRE_WINDOW_SEC),
                kickoff + timedelta(seconds=MATCH_POST_WINDOW_SEC),
            )
        )
    return _merge_windows(windows)


def _seconds_overlap_with_windows(start, end, windows):
    if end <= start:
        return 0.0

    overlap = 0.0
    for w_start, w_end in windows:
        if w_end <= start:
            continue
        if w_start >= end:
            break
        segment_start = max(start, w_start)
        segment_end = min(end, w_end)
        if segment_end > segment_start:
            overlap += (segment_end - segment_start).total_seconds()
    return overlap


def _is_in_active_match_window(now_utc):
    for start, end in _schedule_windows_utc:
        if start <= now_utc <= end:
            return True
    return False


def _choose_base_interval(selection, in_active_window=False):
    if in_active_window:
        if selection == "live":
            return 45
        return 90

    if selection == "live":
        return BASE_INTERVAL_LIVE_SEC
    if selection == "latest_finished":
        return BASE_INTERVAL_FINISHED_SEC
    return BASE_INTERVAL_NONE_SEC


def _adaptive_interval_sec(selection, now_utc=None):
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    in_active_window = _is_in_active_match_window(now_utc)
    interval = float(_choose_base_interval(selection, in_active_window=in_active_window))

    daily_remaining = _rate_limit_state.get("daily_remaining")
    if daily_remaining is not None:
        seconds_left = _seconds_until_utc_day_end(now_utc)
        usable_remaining = max(1, daily_remaining - DAILY_REQUEST_RESERVE)

        day_end = datetime(now_utc.year, now_utc.month, now_utc.day, tzinfo=timezone.utc) + timedelta(days=1)
        active_seconds = _seconds_overlap_with_windows(now_utc, day_end, _schedule_windows_utc)
        idle_seconds = max(0.0, seconds_left - active_seconds)

        active_alloc = max(1, int(usable_remaining * ACTIVE_BUDGET_SHARE))
        idle_alloc = max(1, usable_remaining - active_alloc)

        if in_active_window and active_seconds > 0.0:
            daily_budget_interval = active_seconds / active_alloc
        elif idle_seconds > 0.0:
            daily_budget_interval = idle_seconds / idle_alloc
        else:
            daily_budget_interval = seconds_left / usable_remaining

        interval = max(interval, daily_budget_interval)

    minute_remaining = _rate_limit_state.get("minute_remaining")
    if minute_remaining is not None:
        if minute_remaining <= 1:
            interval = max(interval, 60.0)
        else:
            minute_budget_interval = 60.0 / max(1, minute_remaining - 1)
            interval = max(interval, minute_budget_interval)

    return max(MIN_RETRY_SEC, min(MAX_RETRY_SEC, interval))


def _rate_limit_payload():
    return {
        "requests_limit": _rate_limit_state.get("daily_limit"),
        "requests_remaining": _rate_limit_state.get("daily_remaining"),
        "minute_limit": _rate_limit_state.get("minute_limit"),
        "minute_remaining": _rate_limit_state.get("minute_remaining"),
    }


def _with_rate_limit(payload):
    out = dict(payload)
    out["rate_limit"] = _rate_limit_payload()
    return out


def _refresh_schedule_if_needed(context, now_mono, now_utc):
    global _next_schedule_refresh_after_mono
    global _schedule_windows_utc

    if _schedule_windows_utc and now_mono < _next_schedule_refresh_after_mono:
        return

    if context is None:
        _schedule_windows_utc = []
        _next_schedule_refresh_after_mono = now_mono + SCHEDULE_REFRESH_SEC
        return

    league_id = context.get("league_id")
    seasons = context.get("seasons") or []
    season = seasons[0] if seasons else now_utc.year

    try:
        payload = _fetch_json("fixtures", params={"league": league_id, "season": season})
        fixtures = payload.get("response") or []
        _schedule_windows_utc = _schedule_windows_from_fixtures(fixtures)
        _next_schedule_refresh_after_mono = now_mono + SCHEDULE_REFRESH_SEC
    except requests.RequestException:
        # Retry later without hammering on transient errors.
        _next_schedule_refresh_after_mono = now_mono + (60 * 60)


def _event_datetime(event):
    fixture = event.get("fixture") or {}
    date_str = fixture.get("date")
    if date_str:
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            pass

    ts = fixture.get("timestamp")
    if ts is not None:
        try:
            return datetime.fromtimestamp(int(ts), tz=timezone.utc)
        except (TypeError, ValueError, OSError):
            return None

    return None


def _status_bucket(event):
    fixture = event.get("fixture") or {}
    status = ((fixture.get("status") or {}).get("short") or "").upper()
    if status in {"FT", "AET", "FT_PEN", "PEN", "CANC", "ABD", "AWD", "WO"}:
        return "finished"
    if status in {"NS", "TBD", "PST"}:
        return "scheduled"
    if status:
        return "live"

    goals = event.get("goals") or {}
    home = _safe_int(goals.get("home"))
    away = _safe_int(goals.get("away"))
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
    fixture = event.get("fixture") or {}
    league = event.get("league") or {}
    teams = event.get("teams") or {}
    goals = event.get("goals") or {}
    score = event.get("score") or {}
    status_info = fixture.get("status") or {}

    home_team = ((teams.get("home") or {}).get("name") or "").upper()
    away_team = ((teams.get("away") or {}).get("name") or "").upper()
    home_score = _safe_int(goals.get("home"))
    away_score = _safe_int(goals.get("away"))

    penalties = score.get("penalty") or {}
    home_penalty_score = _safe_int(penalties.get("home"))
    away_penalty_score = _safe_int(penalties.get("away"))

    status = (status_info.get("short") or "NS").upper()
    elapsed = _safe_int(status_info.get("elapsed"))
    extra = _safe_int(status_info.get("extra"))
    minute = ""
    if elapsed is not None:
        minute = str(elapsed)
        if extra:
            minute = f"{minute}+{extra}"

    dt_utc = _event_datetime(event)

    return {
        "event_id": fixture.get("id"),
        "league": league.get("name"),
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
    global _next_status_probe_after_mono

    api_key = os.getenv("API_FOOTBALL_API_KEY") or os.getenv("APIFOOTBALL_API_KEY")
    if not api_key:
        raise RuntimeError("Missing API_FOOTBALL_API_KEY")

    parsed = urlparse(BASE_URL)
    host = parsed.netloc
    headers = {
        "x-apisports-key": api_key,
        "x-apisports-host": host,
    }

    url = f"{BASE_URL}/{path.lstrip('/')}"
    response = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT_SEC)
    response.raise_for_status()
    _update_rate_limit_state(response.headers)

    # Some API-Football endpoints may omit rate headers. Probe /status at low frequency.
    now_mono = time.monotonic()
    if (
        path != "status"
        and _rate_limit_state.get("daily_limit") is None
        and now_mono >= _next_status_probe_after_mono
    ):
        try:
            status_response = requests.get(
                f"{BASE_URL}/status",
                headers=headers,
                timeout=REQUEST_TIMEOUT_SEC,
            )
            status_response.raise_for_status()
            _update_rate_limit_state(status_response.headers)
        except requests.RequestException:
            pass
        finally:
            _next_status_probe_after_mono = now_mono + STATUS_PROBE_COOLDOWN_SEC

    return response.json()


def _get_worldcup_context():
    global _cached_worldcup_context
    if _cached_worldcup_context is False:
        return None
    if _cached_worldcup_context is not None:
        return _cached_worldcup_context

    payload = _fetch_json("leagues", params={"search": LEAGUE_NAME})
    response = payload.get("response") or []
    selected = None
    for row in response:
        league = row.get("league") or {}
        country = row.get("country") or {}
        name = (league.get("name") or "").strip().lower()
        country_name = (country.get("name") or "").strip().lower()
        if name == "world cup" and country_name in {"world", "international"}:
            selected = row
            break

    if selected is None:
        for row in response:
            league = row.get("league") or {}
            country = row.get("country") or {}
            name = (league.get("name") or "").strip().lower()
            country_name = (country.get("name") or "").strip().lower()
            if "world cup" in name and country_name in {"world", "international"}:
                selected = row
                break

    if selected is None:
        _cached_worldcup_context = False
        return None

    league = selected.get("league") or {}
    seasons = selected.get("seasons") or []
    season_years = sorted(
        {
            s.get("year")
            for s in seasons
            if isinstance(s.get("year"), int)
        },
        reverse=True,
    )
    if not season_years:
        _cached_worldcup_context = False
        return None

    _cached_worldcup_context = {
        "league_id": league.get("id"),
        "seasons": season_years,
    }
    return _cached_worldcup_context


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

    global _cached_scorecard
    global _next_fetch_after_mono
    global _next_fallback_lookup_after_mono

    now_mono = time.monotonic()
    now_utc = datetime.now(timezone.utc)
    if _cached_scorecard is not None and now_mono < _next_fetch_after_mono:
        return _with_rate_limit(_cached_scorecard)

    try:
        events = []
        seen_event_ids = set()

        context = _get_worldcup_context()
        _refresh_schedule_if_needed(context, now_mono, now_utc)
        if context is None:
            logger.warning("World Cup competition was not found on API-Football")
            _cached_scorecard = {
                "selected": None,
                "selection": "none",
                "events": [],
            }
            _next_fetch_after_mono = now_mono + _adaptive_interval_sec("none", now_utc=now_utc)
            return _with_rate_limit(_cached_scorecard)

        league_id = context.get("league_id")
        seasons = context.get("seasons") or []
        primary_season = seasons[0] if seasons else datetime.now(timezone.utc).year

        live_payload = _fetch_json("fixtures", params={"league": league_id, "live": "all"})
        for fixture in live_payload.get("response") or []:
            normalized = _normalize_event(fixture)
            event_id = normalized.get("event_id")
            if event_id and event_id in seen_event_ids:
                continue
            if event_id:
                seen_event_ids.add(event_id)
            events.append(normalized)

        selected_live = _choose_live(events)
        if selected_live is not None:
            _cached_scorecard = {
                "selected": selected_live,
                "selection": "live",
                "events": events,
            }
            _next_fetch_after_mono = now_mono + _adaptive_interval_sec("live", now_utc=now_utc)
            return _with_rate_limit(_cached_scorecard)

        today = now_utc.date()
        near_window_start = (today - timedelta(days=1)).isoformat()
        near_window_end = (today + timedelta(days=1)).isoformat()
        if _schedule_windows_utc:
            window_payload = _fetch_json(
                "fixtures",
                params={
                    "league": league_id,
                    "season": primary_season,
                    "from": near_window_start,
                    "to": near_window_end,
                },
            )

            for fixture in window_payload.get("response") or []:
                normalized = _normalize_event(fixture)
                event_id = normalized.get("event_id")
                if event_id and event_id in seen_event_ids:
                    continue
                if event_id:
                    seen_event_ids.add(event_id)
                events.append(normalized)

        if not events and now_mono >= _next_fallback_lookup_after_mono:
            fallback_payload = _fetch_json(
                "fixtures",
                params={"league": league_id, "season": primary_season, "last": 20},
            )
            _next_fallback_lookup_after_mono = now_mono + FALLBACK_LOOKUP_COOLDOWN_SEC
            for fixture in fallback_payload.get("response") or []:
                normalized = _normalize_event(fixture)
                event_id = normalized.get("event_id")
                if event_id and event_id in seen_event_ids:
                    continue
                if event_id:
                    seen_event_ids.add(event_id)
                events.append(normalized)

        if not events:
            _cached_scorecard = {
                "selected": None,
                "selection": "none",
                "events": [],
            }
            _next_fetch_after_mono = now_mono + _adaptive_interval_sec("none", now_utc=now_utc)
            return _with_rate_limit(_cached_scorecard)

        selected_finished = _choose_latest_finished(events)
        if selected_finished is not None:
            _cached_scorecard = {
                "selected": selected_finished,
                "selection": "latest_finished",
                "events": events,
            }
            _next_fetch_after_mono = now_mono + _adaptive_interval_sec("latest_finished", now_utc=now_utc)
            return _with_rate_limit(_cached_scorecard)

        _cached_scorecard = {
            "selected": None,
            "selection": "none",
            "events": events,
        }
        _next_fetch_after_mono = now_mono + _adaptive_interval_sec("none", now_utc=now_utc)
        return _with_rate_limit(_cached_scorecard)
    except requests.RequestException as exc:
        logger.warning("World Cup API request failed: %s", exc)
        _cached_scorecard = {
            "selected": None,
            "selection": "none",
            "events": [],
            "error": f"API request failed: {exc}",
        }
        _next_fetch_after_mono = now_mono + MIN_RETRY_SEC
        return _with_rate_limit(_cached_scorecard)
    except Exception as exc:  # pragma: no cover - defensive runtime safeguard
        logger.exception("Unexpected world cup service failure")
        _cached_scorecard = {
            "selected": None,
            "selection": "none",
            "events": [],
            "error": str(exc),
        }
        _next_fetch_after_mono = now_mono + MIN_RETRY_SEC
        return _with_rate_limit(_cached_scorecard)