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
    # transition_policy binds human_pose via `import app.services.human_pose as
    # human_pose`, i.e. attribute access on the package, so stub both sys.modules
    # and the package attribute. Patching only sys.modules leaks the real module
    # whenever it was imported earlier in the suite.
    services_pkg = importlib.import_module("app.services")

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
    monkeypatch.setattr(services_pkg, "human_pose", human_pose_stub, raising=False)
    monkeypatch.setattr(services_pkg, "worldcup", worldcup_stub, raising=False)
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


class _FaceMeshResults:
    """Minimal face-mesh results stand-in with one detected face."""

    def __init__(self, has_face=True):
        self.multi_face_landmarks = [object()] if has_face else None


def _apply_policy(policy, manager, pose_results=None):
    return policy.apply(
        frame=None,
        pose_results=pose_results,
        mode_manager=manager,
        paint_mode=_Paint(),
        script_mode=_Scripts(),
    )


def _pose_with_landmarks():
    """Minimal pose results stand-in with a detected person."""
    return types.SimpleNamespace(pose_landmarks=object())


def _set_distance(monkeypatch, module, value):
    monkeypatch.setattr(module.human_pose, "estimate_distance", lambda pose_results: (value, []))


def _set_facing(monkeypatch, module, facing=True):
    monkeypatch.setattr(
        module.human_pose,
        "eyes_visible_and_facing_camera",
        lambda pose_results: (facing, "", 5.0),
    )


def _set_monotonic(monkeypatch, module, start=100.0):
    """Fake ``time.monotonic`` in the policy module; returns the mutable clock."""
    mono = {"value": start}
    monkeypatch.setattr(module.time, "monotonic", lambda: mono["value"])
    return mono


def test_caricature_face_keeps_mode_alive_and_supplies_mesh(monkeypatch):
    module, policy = _make_policy(monkeypatch)
    _set_now(monkeypatch, module, datetime(2026, 6, 13, 12, 30, 0))
    manager = ModeManager(mode=ModeManager.MODE_CARICATURE)
    results = _FaceMeshResults()
    monkeypatch.setattr(module.human_pose, "get_face_mesh", lambda frame: results)
    # Stale well past the timeout: a present face must still keep the mode alive.
    manager.mode_update_time = time.time() - (policy.CARICATURE_NO_FACE_TIMEOUT + 5.0)

    state = _apply_policy(policy, manager)

    assert manager.mode == ModeManager.MODE_CARICATURE
    assert state.face_mesh_results is results
    assert manager.get_time_since_last_mode_update() < 1.0


def test_caricature_without_face_returns_to_clock_after_timeout(monkeypatch):
    module, policy = _make_policy(monkeypatch)
    _set_now(monkeypatch, module, datetime(2026, 6, 13, 12, 30, 0))
    manager = ModeManager(mode=ModeManager.MODE_CARICATURE)
    monkeypatch.setattr(module.human_pose, "get_face_mesh", lambda frame: None)

    # Within the grace window: stays in caricature (mesh results may lag entry).
    manager.mode_update_time = time.time() - 1.0
    _apply_policy(policy, manager)
    assert manager.mode == ModeManager.MODE_CARICATURE

    manager.mode_update_time = time.time() - (policy.CARICATURE_NO_FACE_TIMEOUT + 1.0)
    _apply_policy(policy, manager)
    assert manager.mode == ModeManager.MODE_CLOCK


def test_caricature_face_mesh_submit_is_throttled(monkeypatch):
    module, policy = _make_policy(monkeypatch)
    _set_now(monkeypatch, module, datetime(2026, 6, 13, 12, 30, 0))
    manager = ModeManager(mode=ModeManager.MODE_CARICATURE)
    calls = {"count": 0}

    def fake_get_face_mesh(frame):
        calls["count"] += 1
        return _FaceMeshResults()

    monkeypatch.setattr(module.human_pose, "get_face_mesh", fake_get_face_mesh)
    mono = {"value": 100.0}
    monkeypatch.setattr(module.time, "monotonic", lambda: mono["value"])

    _apply_policy(policy, manager)
    _apply_policy(policy, manager)  # same instant: served from cache
    assert calls["count"] == 1

    mono["value"] += policy.face_mesh_submit_interval + 0.01
    _apply_policy(policy, manager)
    assert calls["count"] == 2


def test_caricature_arms_crossed_opens_menu(monkeypatch):
    module, policy = _make_policy(monkeypatch)
    _set_now(monkeypatch, module, datetime(2026, 6, 13, 12, 30, 0))
    manager = ModeManager(mode=ModeManager.MODE_CARICATURE)
    monkeypatch.setattr(module.human_pose, "get_face_mesh", lambda frame: None)
    monkeypatch.setattr(module.human_pose, "is_arms_crossed", lambda pose_results: True)
    clicks = {"count": 0}
    monkeypatch.setattr(
        manager,
        "click_menu",
        lambda entered_via=None: clicks.__setitem__("count", clicks["count"] + 1),
    )

    _apply_policy(policy, manager)

    assert clicks["count"] == 1


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


def test_clock_switches_to_sandfall_when_person_detected(monkeypatch):
    module, policy = _make_policy(monkeypatch)
    _set_now(monkeypatch, module, datetime(2026, 6, 13, 12, 30, 0))
    manager = ModeManager(mode=ModeManager.MODE_CLOCK)
    _set_distance(monkeypatch, module, 1.0)
    _set_facing(monkeypatch, module)

    _apply_policy(policy, manager, pose_results=_pose_with_landmarks())

    assert manager.mode == ModeManager.MODE_SANDFALL
    assert policy._sandfall_via_chain is True


def test_clock_ignores_person_not_facing(monkeypatch):
    module, policy = _make_policy(monkeypatch)
    _set_now(monkeypatch, module, datetime(2026, 6, 13, 12, 30, 0))
    manager = ModeManager(mode=ModeManager.MODE_CLOCK)
    _set_distance(monkeypatch, module, 1.0)
    _set_facing(monkeypatch, module, facing=False)

    _apply_policy(policy, manager, pose_results=_pose_with_landmarks())

    assert manager.mode == ModeManager.MODE_CLOCK


def test_clock_ignores_person_when_pose_disabled(monkeypatch):
    module, policy = _make_policy(monkeypatch)
    _set_now(monkeypatch, module, datetime(2026, 6, 13, 12, 30, 0))
    manager = ModeManager(mode=ModeManager.MODE_CLOCK)
    manager.pose_enabled = False
    _set_distance(monkeypatch, module, 1.0)
    _set_facing(monkeypatch, module)

    _apply_policy(policy, manager, pose_results=_pose_with_landmarks())

    assert manager.mode == ModeManager.MODE_CLOCK


def test_disabling_pose_ends_chain_sandfall(monkeypatch):
    module, policy = _make_policy(monkeypatch)
    _set_now(monkeypatch, module, datetime(2026, 6, 13, 12, 30, 0))
    manager = ModeManager(mode=ModeManager.MODE_SANDFALL)
    policy._sandfall_via_chain = True
    manager.pose_enabled = False
    _set_distance(monkeypatch, module, 1.0)

    _apply_policy(policy, manager, pose_results=_pose_with_landmarks())

    assert manager.mode == ModeManager.MODE_CLOCK
    assert policy._sandfall_via_chain is False


def test_disabling_pose_ends_auto_caricature(monkeypatch):
    module, policy = _make_policy(monkeypatch)
    _set_now(monkeypatch, module, datetime(2026, 6, 13, 12, 30, 0))
    manager = ModeManager(mode=ModeManager.MODE_CARICATURE)
    policy._caricature_return_mode = ModeManager.MODE_SANDFALL
    policy._sandfall_via_chain = True
    manager.pose_enabled = False
    monkeypatch.setattr(module.human_pose, "get_face_mesh", lambda frame: _FaceMeshResults())

    _apply_policy(policy, manager, pose_results=_pose_with_landmarks())

    assert manager.mode == ModeManager.MODE_CLOCK
    assert policy._caricature_return_mode is None
    assert policy._sandfall_via_chain is False


def test_menu_launched_sandfall_survives_pose_disabled(monkeypatch):
    module, policy = _make_policy(monkeypatch)
    _set_now(monkeypatch, module, datetime(2026, 6, 13, 12, 30, 0))
    manager = ModeManager(mode=ModeManager.MODE_SANDFALL)
    manager.pose_enabled = False

    _apply_policy(policy, manager)

    assert manager.mode == ModeManager.MODE_SANDFALL


def test_pose_mode_keepalive_and_timeout(monkeypatch):
    module, policy = _make_policy(monkeypatch)
    _set_now(monkeypatch, module, datetime(2026, 6, 13, 12, 30, 0))
    manager = ModeManager(mode=ModeManager.MODE_POSE)
    manager.mode_update_time = time.time() - 100.0

    _apply_policy(policy, manager, pose_results=_pose_with_landmarks())
    assert manager.mode == ModeManager.MODE_POSE
    assert manager.get_time_since_last_mode_update() < 1.0

    manager.mode_update_time = time.time() - (policy.pose_timeout + 1.0)
    _apply_policy(policy, manager)
    assert manager.mode == ModeManager.MODE_CLOCK


def test_sandfall_supplies_face_mesh_when_close(monkeypatch):
    module, policy = _make_policy(monkeypatch)
    _set_now(monkeypatch, module, datetime(2026, 6, 13, 12, 30, 0))
    manager = ModeManager(mode=ModeManager.MODE_SANDFALL)
    results = _FaceMeshResults()
    monkeypatch.setattr(module.human_pose, "get_face_mesh", lambda frame: results)
    monkeypatch.setattr(module.human_pose, "should_draw_face_features", lambda distance: True)

    state = _apply_policy(policy, manager)

    assert state.face_mesh_results is results


def test_chain_sandfall_very_close_without_facing_never_enters_caricature(monkeypatch):
    module, policy = _make_policy(monkeypatch)
    _set_now(monkeypatch, module, datetime(2026, 6, 13, 12, 30, 0))
    mono = _set_monotonic(monkeypatch, module)
    manager = ModeManager(mode=ModeManager.MODE_SANDFALL)
    policy._sandfall_via_chain = True
    # A turned-away viewer produces wild low readings; they must not trigger.
    _set_distance(monkeypatch, module, 0.4)
    _set_facing(monkeypatch, module, facing=False)

    _apply_policy(policy, manager, pose_results=_pose_with_landmarks())
    mono["value"] += policy.CARICATURE_ENTER_HOLD_SECONDS + 5.0
    _apply_policy(policy, manager, pose_results=_pose_with_landmarks())

    assert manager.mode == ModeManager.MODE_SANDFALL


def test_chain_sandfall_close_blip_does_not_enter_caricature(monkeypatch):
    module, policy = _make_policy(monkeypatch)
    _set_now(monkeypatch, module, datetime(2026, 6, 13, 12, 30, 0))
    mono = _set_monotonic(monkeypatch, module)
    manager = ModeManager(mode=ModeManager.MODE_SANDFALL)
    policy._sandfall_via_chain = True
    _set_facing(monkeypatch, module)

    # Close frame starts the hold, a far frame resets it, close again restarts.
    _set_distance(monkeypatch, module, 0.4)
    _apply_policy(policy, manager, pose_results=_pose_with_landmarks())
    mono["value"] += 0.5
    _set_distance(monkeypatch, module, 1.0)
    _apply_policy(policy, manager, pose_results=_pose_with_landmarks())
    mono["value"] += 0.7
    _set_distance(monkeypatch, module, 0.4)
    _apply_policy(policy, manager, pose_results=_pose_with_landmarks())

    assert manager.mode == ModeManager.MODE_SANDFALL


def test_chain_sandfall_enters_caricature_after_sustained_very_close(monkeypatch):
    module, policy = _make_policy(monkeypatch)
    _set_now(monkeypatch, module, datetime(2026, 6, 13, 12, 30, 0))
    mono = _set_monotonic(monkeypatch, module)
    manager = ModeManager(mode=ModeManager.MODE_SANDFALL)
    policy._sandfall_via_chain = True
    _set_distance(monkeypatch, module, 0.4)
    _set_facing(monkeypatch, module)

    _apply_policy(policy, manager, pose_results=_pose_with_landmarks())
    assert manager.mode == ModeManager.MODE_SANDFALL
    mono["value"] += policy.CARICATURE_ENTER_HOLD_SECONDS + 0.1
    _apply_policy(policy, manager, pose_results=_pose_with_landmarks())

    assert manager.mode == ModeManager.MODE_CARICATURE
    assert policy._caricature_return_mode == ModeManager.MODE_SANDFALL
    # The chain flag survives the hop so the round trip stays a chain.
    assert policy._sandfall_via_chain is True


def test_auto_caricature_returns_to_origin_after_sustained_backing_away(monkeypatch):
    module, policy = _make_policy(monkeypatch)
    _set_now(monkeypatch, module, datetime(2026, 6, 13, 12, 30, 0))
    mono = _set_monotonic(monkeypatch, module)
    manager = ModeManager(mode=ModeManager.MODE_CARICATURE)
    policy._caricature_return_mode = ModeManager.MODE_SANDFALL
    policy._sandfall_via_chain = True
    monkeypatch.setattr(module.human_pose, "get_face_mesh", lambda frame: _FaceMeshResults())
    _set_distance(monkeypatch, module, 0.7)

    # First backing-away frame only starts the hold.
    _apply_policy(policy, manager, pose_results=_pose_with_landmarks())
    assert manager.mode == ModeManager.MODE_CARICATURE

    mono["value"] += policy.CARICATURE_EXIT_HOLD_SECONDS + 0.1
    _apply_policy(policy, manager, pose_results=_pose_with_landmarks())

    assert manager.mode == ModeManager.MODE_SANDFALL
    assert policy._caricature_return_mode is None


def test_auto_caricature_far_blip_does_not_exit(monkeypatch):
    module, policy = _make_policy(monkeypatch)
    _set_now(monkeypatch, module, datetime(2026, 6, 13, 12, 30, 0))
    mono = _set_monotonic(monkeypatch, module)
    manager = ModeManager(mode=ModeManager.MODE_CARICATURE)
    policy._caricature_return_mode = ModeManager.MODE_SANDFALL
    monkeypatch.setattr(module.human_pose, "get_face_mesh", lambda frame: _FaceMeshResults())

    # Far frame starts the hold, a close frame resets it, far again restarts.
    _set_distance(monkeypatch, module, 0.7)
    _apply_policy(policy, manager, pose_results=_pose_with_landmarks())
    mono["value"] += 1.5
    _set_distance(monkeypatch, module, 0.4)
    _apply_policy(policy, manager, pose_results=_pose_with_landmarks())
    mono["value"] += 1.0
    _set_distance(monkeypatch, module, 0.7)
    _apply_policy(policy, manager, pose_results=_pose_with_landmarks())

    assert manager.mode == ModeManager.MODE_CARICATURE
    assert policy._caricature_return_mode == ModeManager.MODE_SANDFALL


def test_auto_caricature_exit_hold_reports_progress(monkeypatch):
    module, policy = _make_policy(monkeypatch)
    _set_now(monkeypatch, module, datetime(2026, 6, 13, 12, 30, 0))
    mono = _set_monotonic(monkeypatch, module)
    manager = ModeManager(mode=ModeManager.MODE_CARICATURE)
    policy._caricature_return_mode = ModeManager.MODE_SANDFALL
    monkeypatch.setattr(module.human_pose, "get_face_mesh", lambda frame: _FaceMeshResults())
    _set_distance(monkeypatch, module, 0.7)

    state = _apply_policy(policy, manager, pose_results=_pose_with_landmarks())
    assert state.caricature_exit_progress == 0.0

    mono["value"] += policy.CARICATURE_EXIT_HOLD_SECONDS / 2.0
    state = _apply_policy(policy, manager, pose_results=_pose_with_landmarks())
    assert state.caricature_exit_progress == 0.5
    assert manager.mode == ModeManager.MODE_CARICATURE

    # An aborted hold stops reporting progress.
    _set_distance(monkeypatch, module, 0.4)
    state = _apply_policy(policy, manager, pose_results=_pose_with_landmarks())
    assert state.caricature_exit_progress is None


def test_auto_caricature_holds_in_hysteresis_dead_band(monkeypatch):
    module, policy = _make_policy(monkeypatch)
    _set_now(monkeypatch, module, datetime(2026, 6, 13, 12, 30, 0))
    mono = _set_monotonic(monkeypatch, module)
    manager = ModeManager(mode=ModeManager.MODE_CARICATURE)
    policy._caricature_return_mode = ModeManager.MODE_SANDFALL
    monkeypatch.setattr(module.human_pose, "get_face_mesh", lambda frame: _FaceMeshResults())
    _set_distance(monkeypatch, module, 0.6)

    _apply_policy(policy, manager, pose_results=_pose_with_landmarks())
    mono["value"] += policy.CARICATURE_EXIT_HOLD_SECONDS + 5.0
    _apply_policy(policy, manager, pose_results=_pose_with_landmarks())

    assert manager.mode == ModeManager.MODE_CARICATURE
    assert policy._caricature_return_mode == ModeManager.MODE_SANDFALL
    assert policy._backing_away_since is None


def test_auto_caricature_no_face_timeout_falls_back_to_clock(monkeypatch):
    module, policy = _make_policy(monkeypatch)
    _set_now(monkeypatch, module, datetime(2026, 6, 13, 12, 30, 0))
    manager = ModeManager(mode=ModeManager.MODE_CARICATURE)
    policy._caricature_return_mode = ModeManager.MODE_SANDFALL
    monkeypatch.setattr(module.human_pose, "get_face_mesh", lambda frame: None)
    manager.mode_update_time = time.time() - (policy.CARICATURE_NO_FACE_TIMEOUT + 1.0)

    _apply_policy(policy, manager)

    assert manager.mode == ModeManager.MODE_CLOCK
    assert policy._caricature_return_mode is None


def test_manual_caricature_ignores_distance(monkeypatch):
    module, policy = _make_policy(monkeypatch)
    _set_now(monkeypatch, module, datetime(2026, 6, 13, 12, 30, 0))
    manager = ModeManager(mode=ModeManager.MODE_CARICATURE)
    monkeypatch.setattr(module.human_pose, "get_face_mesh", lambda frame: _FaceMeshResults())
    _set_distance(monkeypatch, module, 2.0)

    state = _apply_policy(policy, manager, pose_results=_pose_with_landmarks())

    assert manager.mode == ModeManager.MODE_CARICATURE
    # Manual caricature never runs the exit hold, so no shrink is reported.
    assert state.caricature_exit_progress is None


def test_chain_sandfall_returns_to_clock_when_person_leaves(monkeypatch):
    module, policy = _make_policy(monkeypatch)
    _set_now(monkeypatch, module, datetime(2026, 6, 13, 12, 30, 0))
    manager = ModeManager(mode=ModeManager.MODE_SANDFALL)
    policy._sandfall_via_chain = True
    manager.mode_update_time = time.time() - (policy.pose_timeout + 1.0)

    _apply_policy(policy, manager)

    assert manager.mode == ModeManager.MODE_CLOCK


def test_chain_sandfall_keepalive_with_person_present(monkeypatch):
    module, policy = _make_policy(monkeypatch)
    _set_now(monkeypatch, module, datetime(2026, 6, 13, 12, 30, 0))
    manager = ModeManager(mode=ModeManager.MODE_SANDFALL)
    policy._sandfall_via_chain = True
    manager.mode_update_time = time.time() - (policy.pose_timeout + 1.0)
    _set_distance(monkeypatch, module, 1.0)

    _apply_policy(policy, manager, pose_results=_pose_with_landmarks())

    assert manager.mode == ModeManager.MODE_SANDFALL
    assert manager.get_time_since_last_mode_update() < 1.0


def test_menu_launched_sandfall_idles_without_person(monkeypatch):
    module, policy = _make_policy(monkeypatch)
    _set_now(monkeypatch, module, datetime(2026, 6, 13, 12, 30, 0))
    manager = ModeManager(mode=ModeManager.MODE_SANDFALL)
    manager.mode_update_time = time.time() - 100.0

    _apply_policy(policy, manager)

    assert manager.mode == ModeManager.MODE_SANDFALL


def test_menu_launched_sandfall_ignores_very_close_person(monkeypatch):
    module, policy = _make_policy(monkeypatch)
    _set_now(monkeypatch, module, datetime(2026, 6, 13, 12, 30, 0))
    mono = _set_monotonic(monkeypatch, module)
    manager = ModeManager(mode=ModeManager.MODE_SANDFALL)
    _set_distance(monkeypatch, module, 0.4)
    _set_facing(monkeypatch, module)

    _apply_policy(policy, manager, pose_results=_pose_with_landmarks())
    mono["value"] += policy.CARICATURE_ENTER_HOLD_SECONDS + 5.0
    _apply_policy(policy, manager, pose_results=_pose_with_landmarks())

    assert manager.mode == ModeManager.MODE_SANDFALL


def test_sandfall_arms_crossed_opens_menu(monkeypatch):
    module, policy = _make_policy(monkeypatch)
    _set_now(monkeypatch, module, datetime(2026, 6, 13, 12, 30, 0))
    manager = ModeManager(mode=ModeManager.MODE_SANDFALL)
    monkeypatch.setattr(module.human_pose, "is_arms_crossed", lambda pose_results: True)
    clicks = {"count": 0}
    monkeypatch.setattr(
        manager,
        "click_menu",
        lambda entered_via=None: clicks.__setitem__("count", clicks["count"] + 1),
    )

    _apply_policy(policy, manager)

    assert clicks["count"] == 1


def test_external_mode_change_clears_chain_flags(monkeypatch):
    module, policy = _make_policy(monkeypatch)
    _set_now(monkeypatch, module, datetime(2026, 6, 13, 12, 30, 0))
    manager = ModeManager(mode=ModeManager.MODE_MENU)
    policy._caricature_return_mode = ModeManager.MODE_SANDFALL
    policy._sandfall_via_chain = True

    _apply_policy(policy, manager)

    assert policy._caricature_return_mode is None
    assert policy._sandfall_via_chain is False
