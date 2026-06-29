import importlib
import sys
import time
import types
from datetime import datetime

from app.core.mode_manager import ModeManager


class _Paint:
    def clear(self):
        return None


class _Scripts:
    """Stub script mode recording the calls the transition policy makes."""

    def __init__(self, can_start=False):
        self.can_start = can_start
        self.reshuffle_calls = 0
        self.start_calls = 0
        self.stop_calls = 0

    def reshuffle_day(self):
        self.reshuffle_calls += 1

    def start_next(self):
        self.start_calls += 1
        return self.can_start

    def stop_script(self):
        self.stop_calls += 1
        return True


def _set_now(monkeypatch, module, when):
    monkeypatch.setattr(
        module, "datetime", type("_FakeDatetime", (), {"now": staticmethod(lambda: when)})
    )


def _load_transition_policy_module(monkeypatch):
    importlib.import_module("app.services")

    human_pose_stub = types.SimpleNamespace(
        is_arms_crossed=lambda pose_results: False,
        eyes_visible_and_facing_camera=lambda pose_results: (False, "", None),
        estimate_distance=lambda pose_results: (None, None),
        should_draw_face_features=lambda distance: False,
        get_face_mesh=lambda frame: None,
    )
    worldcup_stub = types.SimpleNamespace(get_worldcup_scorecard=lambda: {"events": []})
    monkeypatch.setitem(sys.modules, "app.services.human_pose", human_pose_stub)
    monkeypatch.setitem(sys.modules, "app.services.worldcup", worldcup_stub)
    sys.modules.pop("app.core.transition_policy", None)
    return importlib.import_module("app.core.transition_policy")


def test_is_sleep_hour_boundaries(monkeypatch):
    transition_policy_module = _load_transition_policy_module(monkeypatch)
    policy = transition_policy_module.TransitionPolicy(
        pose_timeout=3,
        sleep_start_hour=23,
        sleep_end_hour=7,
    )
    assert policy.is_sleep_hour(datetime(2026, 6, 13, 22, 59, 0)) is False
    assert policy.is_sleep_hour(datetime(2026, 6, 13, 23, 0, 0)) is True
    assert policy.is_sleep_hour(datetime(2026, 6, 14, 2, 0, 0)) is True
    assert policy.is_sleep_hour(datetime(2026, 6, 14, 7, 0, 0)) is False

    policy2 = transition_policy_module.TransitionPolicy(
        pose_timeout=3,
        sleep_start_hour=1,
        sleep_end_hour=7,
    )
    assert policy2.is_sleep_hour(datetime(2026, 6, 13, 0, 59, 0)) is False
    assert policy2.is_sleep_hour(datetime(2026, 6, 13, 1, 0, 0)) is True
    assert policy2.is_sleep_hour(datetime(2026, 6, 13, 6, 59, 0)) is True
    assert policy2.is_sleep_hour(datetime(2026, 6, 13, 7, 0, 0)) is False


def test_sleep_settings_can_disable_sleep_mode(monkeypatch):
    transition_policy_module = _load_transition_policy_module(monkeypatch)
    policy = transition_policy_module.TransitionPolicy(
        pose_timeout=3,
        sleep_start_hour=1,
        sleep_end_hour=7,
    )

    assert policy.is_sleep_hour(datetime(2026, 6, 13, 2, 0, 0)) is True

    policy.set_sleep_settings(enabled=False, start_hour=1, end_hour=7)
    assert policy.is_sleep_hour(datetime(2026, 6, 13, 2, 0, 0)) is False


def test_sleep_settings_getter_returns_updated_values(monkeypatch):
    transition_policy_module = _load_transition_policy_module(monkeypatch)
    policy = transition_policy_module.TransitionPolicy(
        pose_timeout=3,
        sleep_start_hour=0,
        sleep_end_hour=7,
    )

    settings = policy.set_sleep_settings(enabled=True, start_hour=23, end_hour=99)

    assert settings == {
        "enabled": True,
        "start_hour": 23,
        "end_hour": 23,
    }
    assert policy.get_sleep_settings() == settings


def test_sleep_window_with_equal_bounds_is_empty(monkeypatch):
    transition_policy_module = _load_transition_policy_module(monkeypatch)
    policy = transition_policy_module.TransitionPolicy(
        pose_timeout=3,
        sleep_start_hour=7,
        sleep_end_hour=7,
    )

    assert policy.is_sleep_hour(datetime(2026, 6, 13, 6, 0, 0)) is False
    assert policy.is_sleep_hour(datetime(2026, 6, 13, 7, 0, 0)) is False
    assert policy.is_sleep_hour(datetime(2026, 6, 13, 23, 0, 0)) is False


def test_sleep_preempts_worldcup_mode(monkeypatch):
    transition_policy_module = _load_transition_policy_module(monkeypatch)
    policy = transition_policy_module.TransitionPolicy(
        pose_timeout=3,
        sleep_start_hour=23,
        sleep_end_hour=7,
    )
    manager = ModeManager(mode=ModeManager.MODE_WORLDCUP)

    monkeypatch.setattr(
        transition_policy_module,
        "datetime",
        type("_FakeDatetime", (), {"now": staticmethod(lambda: datetime(2026, 6, 13, 23, 15, 0))}),
    )

    state = policy.apply(
        frame=None,
        pose_results=None,
        mode_manager=manager,
        paint_mode=_Paint(),
        script_mode=_Scripts(),
    )

    assert manager.mode == ModeManager.MODE_SLEEP
    assert state.reason == ""


def test_disabling_sleep_wakes_from_sleep_mode(monkeypatch):
    transition_policy_module = _load_transition_policy_module(monkeypatch)
    policy = transition_policy_module.TransitionPolicy(
        pose_timeout=3,
        sleep_start_hour=1,
        sleep_end_hour=7,
    )
    manager = ModeManager(mode=ModeManager.MODE_SLEEP)

    policy.set_sleep_settings(enabled=False, start_hour=1, end_hour=7)
    monkeypatch.setattr(
        transition_policy_module,
        "datetime",
        type("_FakeDatetime", (), {"now": staticmethod(lambda: datetime(2026, 6, 13, 2, 0, 0))}),
    )

    policy.apply(
        frame=None,
        pose_results=None,
        mode_manager=manager,
        paint_mode=_Paint(),
        script_mode=_Scripts(),
    )

    assert manager.mode == ModeManager.MODE_CLOCK


def test_worldcup_live_check_uses_cache_interval(monkeypatch):
    transition_policy_module = _load_transition_policy_module(monkeypatch)
    policy = transition_policy_module.TransitionPolicy(
        pose_timeout=3,
        sleep_start_hour=1,
        sleep_end_hour=7,
    )

    calls = {"count": 0}

    def fake_scorecard():
        calls["count"] += 1
        return {"events": [{"status_bucket": "live"}]}

    now = {"value": 31.0}

    def fake_monotonic():
        return now["value"]

    class _ImmediateThread:
        def __init__(self, target, daemon=None):
            self._target = target

        def start(self):
            self._target()

    monkeypatch.setattr(transition_policy_module, "get_worldcup_scorecard", fake_scorecard)
    monkeypatch.setattr(transition_policy_module.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(transition_policy_module.threading, "Thread", _ImmediateThread)

    # First call starts refresh in the background and returns cached value.
    assert policy._is_worldcup_live() is False
    assert calls["count"] == 1

    now["value"] = 40.0
    assert policy._is_worldcup_live() is True
    assert calls["count"] == 1

    now["value"] = 62.0
    assert policy._is_worldcup_live() is True
    assert calls["count"] == 2


class _ImmediateThread:
    def __init__(self, target, daemon=None):
        self._target = target

    def start(self):
        self._target()


def _make_worldcup_policy(monkeypatch, events_ref, now):
    transition_policy_module = _load_transition_policy_module(monkeypatch)
    policy = transition_policy_module.TransitionPolicy(
        pose_timeout=3,
        sleep_start_hour=1,
        sleep_end_hour=7,
    )

    def fake_scorecard():
        return {"events": list(events_ref["value"])}

    monkeypatch.setattr(transition_policy_module, "get_worldcup_scorecard", fake_scorecard)
    monkeypatch.setattr(transition_policy_module.time, "monotonic", lambda: now["value"])
    monkeypatch.setattr(transition_policy_module.threading, "Thread", _ImmediateThread)
    return policy


def _live(event_id):
    return {"status_bucket": "live", "event_id": event_id}


def test_manual_clock_suppresses_switch_until_new_match_goes_live(monkeypatch):
    events_ref = {"value": [_live(101)]}
    now = {"value": 100.0}
    policy = _make_worldcup_policy(monkeypatch, events_ref, now)
    manager = ModeManager(mode=ModeManager.MODE_CLOCK)

    # User manually picks clock while match 101 is live: stay on clock.
    manager.note_manual_clock_selection()
    assert policy._should_autoswitch_to_worldcup(manager) is False

    # The same match staying live does not re-trigger the switch.
    now["value"] += 31.0
    assert policy._should_autoswitch_to_worldcup(manager) is False

    # A different match going live counts as new and hands off to World Cup.
    events_ref["value"] = [_live(101), _live(202)]
    now["value"] += 31.0
    assert policy._should_autoswitch_to_worldcup(manager) is True


def test_autoswitch_fires_without_manual_clock_selection(monkeypatch):
    events_ref = {"value": [_live(101)]}
    now = {"value": 100.0}
    policy = _make_worldcup_policy(monkeypatch, events_ref, now)
    manager = ModeManager(mode=ModeManager.MODE_CLOCK)

    # Clock entered by policy (no manual selection): a live match switches.
    assert policy._should_autoswitch_to_worldcup(manager) is True


def test_new_match_after_acknowledged_one_finishes(monkeypatch):
    events_ref = {"value": [_live(101)]}
    now = {"value": 100.0}
    policy = _make_worldcup_policy(monkeypatch, events_ref, now)
    manager = ModeManager(mode=ModeManager.MODE_CLOCK)

    manager.note_manual_clock_selection()
    assert policy._should_autoswitch_to_worldcup(manager) is False

    # Match 101 finishes, then a later match goes live: switch.
    events_ref["value"] = [_live(303)]
    now["value"] += 31.0
    assert policy._should_autoswitch_to_worldcup(manager) is True


def _make_policy(monkeypatch):
    module = _load_transition_policy_module(monkeypatch)
    policy = module.TransitionPolicy(pose_timeout=3, sleep_start_hour=1, sleep_end_hour=7)
    return module, policy


def test_hourly_script_starts_at_top_of_hour(monkeypatch):
    module, policy = _make_policy(monkeypatch)
    manager = ModeManager(mode=ModeManager.MODE_CLOCK)
    scripts = _Scripts(can_start=True)
    _set_now(monkeypatch, module, datetime(2026, 6, 13, 14, 0, 0))

    policy.apply(
        frame=None,
        pose_results=None,
        mode_manager=manager,
        paint_mode=_Paint(),
        script_mode=scripts,
    )

    assert manager.mode == ModeManager.MODE_SCRIPT
    assert scripts.start_calls == 1


def test_hourly_script_not_retriggered_same_hour(monkeypatch):
    module, policy = _make_policy(monkeypatch)
    manager = ModeManager(mode=ModeManager.MODE_CLOCK)
    scripts = _Scripts(can_start=True)

    _set_now(monkeypatch, module, datetime(2026, 6, 13, 14, 0, 0))
    policy.apply(
        frame=None,
        pose_results=None,
        mode_manager=manager,
        paint_mode=_Paint(),
        script_mode=scripts,
    )
    assert scripts.start_calls == 1

    # Manually back on the clock in the same hour: no re-trigger.
    manager.set_mode(ModeManager.MODE_CLOCK)
    policy.apply(
        frame=None,
        pose_results=None,
        mode_manager=manager,
        paint_mode=_Paint(),
        script_mode=scripts,
    )
    assert scripts.start_calls == 1
    assert manager.mode == ModeManager.MODE_CLOCK

    # Next hour triggers again.
    _set_now(monkeypatch, module, datetime(2026, 6, 13, 15, 0, 0))
    policy.apply(
        frame=None,
        pose_results=None,
        mode_manager=manager,
        paint_mode=_Paint(),
        script_mode=scripts,
    )
    assert scripts.start_calls == 2
    assert manager.mode == ModeManager.MODE_SCRIPT


def test_hourly_script_returns_to_clock_after_duration(monkeypatch):
    module, policy = _make_policy(monkeypatch)
    manager = ModeManager(mode=ModeManager.MODE_SCRIPT)
    manager.mode_start_time = time.time() - (policy.HOURLY_SCRIPT_DURATION + 1)
    policy._hourly_script_active = True
    scripts = _Scripts(can_start=True)
    _set_now(monkeypatch, module, datetime(2026, 6, 13, 14, 0, 30))

    policy.apply(
        frame=None,
        pose_results=None,
        mode_manager=manager,
        paint_mode=_Paint(),
        script_mode=scripts,
    )

    assert manager.mode == ModeManager.MODE_CLOCK
    assert scripts.stop_calls == 1


def test_hourly_script_empty_library_stays_on_clock(monkeypatch):
    module, policy = _make_policy(monkeypatch)
    manager = ModeManager(mode=ModeManager.MODE_CLOCK)
    scripts = _Scripts(can_start=False)
    _set_now(monkeypatch, module, datetime(2026, 6, 13, 14, 0, 0))

    policy.apply(
        frame=None,
        pose_results=None,
        mode_manager=manager,
        paint_mode=_Paint(),
        script_mode=scripts,
    )

    assert scripts.start_calls == 1
    assert manager.mode == ModeManager.MODE_CLOCK


def test_reshuffle_runs_once_per_day_during_sleep(monkeypatch):
    module, policy = _make_policy(monkeypatch)
    manager = ModeManager(mode=ModeManager.MODE_CLOCK)
    scripts = _Scripts()

    def _apply(when):
        _set_now(monkeypatch, module, when)
        policy.apply(
            frame=None,
            pose_results=None,
            mode_manager=manager,
            paint_mode=_Paint(),
            script_mode=scripts,
        )

    # Reshuffles overnight, even on the frame that enters sleep mode.
    _apply(datetime(2026, 6, 13, 2, 0, 0))
    assert manager.mode == ModeManager.MODE_SLEEP
    assert scripts.reshuffle_calls == 1

    # Same date, still asleep: no second reshuffle.
    _apply(datetime(2026, 6, 13, 3, 0, 0))
    assert scripts.reshuffle_calls == 1

    # New date: reshuffle again.
    _apply(datetime(2026, 6, 14, 2, 0, 0))
    assert scripts.reshuffle_calls == 2


def test_reshuffle_when_sleep_disabled_fires_from_5am(monkeypatch):
    module, policy = _make_policy(monkeypatch)
    policy.set_sleep_settings(enabled=False, start_hour=1, end_hour=7)
    manager = ModeManager(mode=ModeManager.MODE_CLOCK)
    scripts = _Scripts()

    def _apply(when):
        _set_now(monkeypatch, module, when)
        policy.apply(
            frame=None,
            pose_results=None,
            mode_manager=manager,
            paint_mode=_Paint(),
            script_mode=scripts,
        )

    _apply(datetime(2026, 6, 13, 4, 0, 0))
    assert scripts.reshuffle_calls == 0  # before 5 a.m.

    _apply(datetime(2026, 6, 13, 5, 0, 0))
    assert scripts.reshuffle_calls == 1
