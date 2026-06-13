from datetime import datetime
import importlib
import sys
import types


def _load_transition_policy_module(monkeypatch):
    human_pose_stub = types.SimpleNamespace(
        is_arms_crossed=lambda pose_results: False,
        eyes_visible_and_facing_camera=lambda pose_results: (False, "", None),
        estimate_distance=lambda pose_results: (None, None),
        should_draw_face_features=lambda distance: False,
        get_face_mesh=lambda frame: None,
    )
    monkeypatch.setitem(sys.modules, "app.services.human_pose", human_pose_stub)
    sys.modules.pop("app.core.transition_policy", None)
    return importlib.import_module("app.core.transition_policy")


def test_is_sleep_hour_boundaries(monkeypatch):
    transition_policy_module = _load_transition_policy_module(monkeypatch)
    policy = transition_policy_module.TransitionPolicy(
        pose_timeout=3,
        sleep_start_hour=23,
        sleep_end_hour=7,
    )
    # Preserve current non-wrapping behavior.
    assert policy.is_sleep_hour(datetime(2026, 6, 13, 23, 0, 0)) is False
    assert policy.is_sleep_hour(datetime(2026, 6, 13, 2, 0, 0)) is False

    policy2 = transition_policy_module.TransitionPolicy(
        pose_timeout=3,
        sleep_start_hour=1,
        sleep_end_hour=7,
    )
    assert policy2.is_sleep_hour(datetime(2026, 6, 13, 0, 59, 0)) is False
    assert policy2.is_sleep_hour(datetime(2026, 6, 13, 1, 0, 0)) is True
    assert policy2.is_sleep_hour(datetime(2026, 6, 13, 6, 59, 0)) is True
    assert policy2.is_sleep_hour(datetime(2026, 6, 13, 7, 0, 0)) is False


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

    monkeypatch.setattr(transition_policy_module, "get_worldcup_scorecard", fake_scorecard)
    monkeypatch.setattr(transition_policy_module.time, "monotonic", fake_monotonic)

    assert policy._is_worldcup_live() is True
    assert calls["count"] == 1

    now["value"] = 40.0
    assert policy._is_worldcup_live() is True
    assert calls["count"] == 1

    now["value"] = 62.0
    assert policy._is_worldcup_live() is True
    assert calls["count"] == 2
