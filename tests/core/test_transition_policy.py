from datetime import datetime
import importlib
import sys
import types

from app.core.mode_manager import ModeManager


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

    class _Paint:
        def clear(self):
            return None

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

    class _Paint:
        def clear(self):
            return None

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
