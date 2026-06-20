import importlib
import sys
import types


def _load_input_source_module(monkeypatch):
    human_pose_stub = types.SimpleNamespace(
        get_right_index_finger_position=lambda pose_results: (None, None)
    )
    monkeypatch.setitem(sys.modules, "app.services.human_pose", human_pose_stub)
    sys.modules.pop("app.core.input_source", None)
    return importlib.import_module("app.core.input_source")


def test_submit_pointer_clamps_to_unit_interval(monkeypatch):
    input_source_module = _load_input_source_module(monkeypatch)
    monkeypatch.setattr(input_source_module.time, "monotonic", lambda: 10.0)
    hub = input_source_module.InputHub()
    hub.submit_pointer(source="web", x=-0.3, y=1.8, timestamp=10.0)
    sample = hub.get_active_pointer(max_age_sec=9999)

    assert sample is not None
    assert sample.x == 0.0
    assert sample.y == 1.0


def test_get_active_pointer_returns_none_when_stale(monkeypatch):
    input_source_module = _load_input_source_module(monkeypatch)
    now = {"value": 100.0}

    def fake_monotonic():
        return now["value"]

    monkeypatch.setattr(input_source_module.time, "monotonic", fake_monotonic)
    hub = input_source_module.InputHub()
    hub.submit_pointer(source="web", x=0.4, y=0.4, timestamp=100.0)

    now["value"] = 102.0
    assert hub.get_active_pointer(max_age_sec=1.0) is None


def test_get_active_pointer_prefers_newest_sample(monkeypatch):
    input_source_module = _load_input_source_module(monkeypatch)
    now = {"value": 10.0}

    def fake_monotonic():
        return now["value"]

    monkeypatch.setattr(input_source_module.time, "monotonic", fake_monotonic)
    hub = input_source_module.InputHub()

    hub.submit_pointer(source="pose", x=0.1, y=0.2, timestamp=9.0)
    hub.submit_pointer(source="web", x=0.8, y=0.9, timestamp=9.5)
    sample = hub.get_active_pointer(max_age_sec=2.0)

    assert sample is not None
    assert sample.source == "web"
    assert sample.x == 0.8
    assert sample.y == 0.9


def test_get_active_pointer_can_filter_allowed_sources(monkeypatch):
    input_source_module = _load_input_source_module(monkeypatch)
    now = {"value": 10.0}

    def fake_monotonic():
        return now["value"]

    monkeypatch.setattr(input_source_module.time, "monotonic", fake_monotonic)
    hub = input_source_module.InputHub()

    hub.submit_pointer(source="pose", x=0.2, y=0.2, timestamp=9.0)
    hub.submit_pointer(source="controller", x=0.9, y=0.9, timestamp=9.5)

    sample = hub.get_active_pointer(max_age_sec=2.0, allowed_sources={"pose"})

    assert sample is not None
    assert sample.source == "pose"


def test_pop_actions_filters_by_age(monkeypatch):
    input_source_module = _load_input_source_module(monkeypatch)
    now = {"value": 50.0}

    def fake_monotonic():
        return now["value"]

    monkeypatch.setattr(input_source_module.time, "monotonic", fake_monotonic)
    hub = input_source_module.InputHub()
    hub.submit_action(source="web", action="next", timestamp=47.5)
    hub.submit_action(source="web", action="prev", timestamp=49.5)

    actions = hub.pop_actions(max_age_sec=2.0)

    assert [a.action for a in actions] == ["prev"]
    assert hub.pop_actions(max_age_sec=2.0) == []


def test_pop_actions_can_filter_allowed_sources(monkeypatch):
    input_source_module = _load_input_source_module(monkeypatch)
    now = {"value": 50.0}

    def fake_monotonic():
        return now["value"]

    monkeypatch.setattr(input_source_module.time, "monotonic", fake_monotonic)
    hub = input_source_module.InputHub()
    hub.submit_action(source="pose", action="toggle_menu", timestamp=49.5)
    hub.submit_action(source="controller", action="toggle_menu", timestamp=49.5)

    actions = hub.pop_actions(max_age_sec=2.0, allowed_sources={"pose"})

    assert [a.source for a in actions] == ["pose"]


def test_pop_clicks_filters_by_age(monkeypatch):
    input_source_module = _load_input_source_module(monkeypatch)
    now = {"value": 12.0}

    def fake_monotonic():
        return now["value"]

    monkeypatch.setattr(input_source_module.time, "monotonic", fake_monotonic)
    hub = input_source_module.InputHub()
    hub.submit_click(source="web", x=0.1, y=0.2, timestamp=10.5)
    hub.submit_click(source="web", x=0.9, y=0.8, timestamp=11.7)

    clicks = hub.pop_clicks(max_age_sec=1.0)

    assert len(clicks) == 1
    assert clicks[0].x == 0.9
    assert clicks[0].y == 0.8
    assert hub.pop_clicks(max_age_sec=1.0) == []


def test_pop_clicks_can_filter_allowed_sources(monkeypatch):
    input_source_module = _load_input_source_module(monkeypatch)
    now = {"value": 12.0}

    def fake_monotonic():
        return now["value"]

    monkeypatch.setattr(input_source_module.time, "monotonic", fake_monotonic)
    hub = input_source_module.InputHub()
    hub.submit_click(source="pose", x=0.1, y=0.2, timestamp=11.8)
    hub.submit_click(source="controller", x=0.9, y=0.8, timestamp=11.9)

    clicks = hub.pop_clicks(max_age_sec=1.0, allowed_sources={"pose"})

    assert len(clicks) == 1
    assert clicks[0].source == "pose"
