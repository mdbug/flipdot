import importlib
import sys
import types

import numpy as np


def _load_menu_module(monkeypatch):
    human_pose_stub = types.SimpleNamespace(
        get_right_index_finger_position=lambda pose_results: (None, None),
        draw_right_index_pointer=lambda frame, pose_results, size=1: frame,
        draw_pointer=lambda frame, x, y, mirror_x=False: frame,
    )
    mediapipe_stub = types.SimpleNamespace(
        solutions=types.SimpleNamespace(
            pose=types.SimpleNamespace(PoseLandmark=types.SimpleNamespace())
        )
    )

    monkeypatch.setitem(sys.modules, "app.services.human_pose", human_pose_stub)
    monkeypatch.setitem(sys.modules, "mediapipe", mediapipe_stub)
    sys.modules.pop("app.modes.menu", None)
    return importlib.import_module("app.modes.menu")


class DummyModeManager:
    def __init__(self):
        self.pose_enabled = True
        self.modes = []
        self.toggle_count = 0
        self.allowed_sources = {"pose", "controller", "web"}

    def set_mode(self, mode):
        self.modes.append(mode)

    def toggle_pose_enabled(self):
        self.pose_enabled = not self.pose_enabled
        self.toggle_count += 1

    def get_allowed_input_sources(self, include_web=True):
        if include_web:
            return set(self.allowed_sources)
        return {source for source in self.allowed_sources if source != "web"}


class DummyInputHub:
    def __init__(self, pointer=None, clicks=None):
        self._pointer = pointer
        self._clicks = list(clicks or [])

    def get_active_pointer(self, max_age_sec=0.8, allowed_sources=None):
        if self._pointer is None:
            return None
        if allowed_sources is not None and self._pointer.source not in allowed_sources:
            return None
        return self._pointer

    def pop_clicks(self, max_age_sec=1.2, allowed_sources=None):
        clicks = self._clicks
        self._clicks = []
        if allowed_sources is None:
            return clicks
        return [click for click in clicks if click.source in allowed_sources]


def test_menuitem_hover_triggers_click_after_dwell(monkeypatch):
    menu_module = _load_menu_module(monkeypatch)
    now = {"value": 10.0}
    monkeypatch.setattr(menu_module.time, "time", lambda: now["value"])

    calls = {"count": 0}
    item = menu_module.MenuItem(
        "A",
        row=0,
        width=28,
        on_click=lambda source=None: calls.__setitem__("count", calls["count"] + 1),
    )

    item.hover(True)
    assert item.hovered is True

    now["value"] = 12.1
    item.hover(True)

    assert calls["count"] == 1
    assert item.hover_start_time is None


def test_indicator_hover_changes_page_after_dwell(monkeypatch):
    menu_module = _load_menu_module(monkeypatch)
    manager = DummyModeManager()
    menu = menu_module.Menu(width=28, height=28, mode_manager=manager)

    menu._update_indicator_hover(panel_x=10, panel_y=25, now=0.0, hovered_page=1)
    assert menu.page == 0

    menu._update_indicator_hover(panel_x=10, panel_y=25, now=2.1, hovered_page=1)
    assert menu.page == 1


def test_menu_click_on_item_calls_mode_switch(monkeypatch):
    menu_module = _load_menu_module(monkeypatch)
    monkeypatch.setattr(menu_module.time, "time", lambda: 1.0)

    manager = DummyModeManager()
    menu = menu_module.Menu(width=28, height=28, mode_manager=manager)
    click = types.SimpleNamespace(source="web", x=0.10, y=0.05)
    hub = DummyInputHub(pointer=None, clicks=[click])

    frame = menu.get_frame(pose_results=None, input_hub=hub)

    assert isinstance(frame, np.ndarray)
    assert frame.shape == (28, 28)
    assert manager.modes[-1] == menu_module.ModeManager.MODE_CLOCK


def test_pointer_to_panel_mirrors_pose_but_not_web(monkeypatch):
    menu_module = _load_menu_module(monkeypatch)
    manager = DummyModeManager()
    menu = menu_module.Menu(width=28, height=28, mode_manager=manager)

    pose_x, pose_y = menu._pointer_to_panel("pose", 0.25, 0.5)
    web_x, web_y = menu._pointer_to_panel("web", 0.25, 0.5)

    assert pose_x > web_x
    assert pose_y == web_y


def test_swipe_advances_and_reverses_pages(monkeypatch):
    menu_module = _load_menu_module(monkeypatch)
    manager = DummyModeManager()
    menu = menu_module.Menu(width=28, height=28, mode_manager=manager)
    menu._last_swipe_time = -1.0

    menu._update_swipe(5, 10, now=0.00)
    menu._update_swipe(9, 10, now=0.05)
    menu._update_swipe(18, 10, now=0.15)
    assert menu.page == 1

    menu._swipe_locked = False
    menu._swipe_origin = None
    menu._finger_sample = None
    menu._last_swipe_time = 0.0
    menu._last_swipe_direction = 0

    menu._update_swipe(18, 10, now=1.00)
    menu._update_swipe(14, 10, now=1.05)
    menu._update_swipe(5, 10, now=1.15)
    assert menu.page == 0


def test_indicator_click_changes_page_without_item_click(monkeypatch):
    menu_module = _load_menu_module(monkeypatch)
    monkeypatch.setattr(menu_module.time, "time", lambda: 10.0)

    manager = DummyModeManager()
    menu = menu_module.Menu(width=28, height=28, mode_manager=manager)
    click = types.SimpleNamespace(source="web", x=0.90, y=0.95)
    hub = DummyInputHub(pointer=None, clicks=[click])

    menu.get_frame(pose_results=None, input_hub=hub)

    assert menu.page == 4
    assert manager.modes == []


def test_fonts_button_is_on_new_page(monkeypatch):
    menu_module = _load_menu_module(monkeypatch)
    manager = DummyModeManager()
    menu = menu_module.Menu(width=28, height=28, mode_manager=manager)

    assert len(menu.pages) >= 5
    labels = [item.label for item in menu.pages[4]]
    assert "FONTS" in labels


def test_face_button_switches_to_caricature(monkeypatch):
    menu_module = _load_menu_module(monkeypatch)
    manager = DummyModeManager()
    menu = menu_module.Menu(width=28, height=28, mode_manager=manager)

    face = next(item for page in menu.pages for item in page if item.label == "FACE")
    face.click("web")

    assert manager.modes[-1] == menu_module.ModeManager.MODE_CARICATURE


def test_life_and_sand_buttons_switch_modes(monkeypatch):
    menu_module = _load_menu_module(monkeypatch)
    manager = DummyModeManager()
    menu = menu_module.Menu(width=28, height=28, mode_manager=manager)

    life = next(item for page in menu.pages for item in page if item.label == "LIFE")
    life.click("web")
    sand = next(item for page in menu.pages for item in page if item.label == "SAND")
    sand.click("web")

    assert menu_module.ModeManager.MODE_LIFE in manager.modes
    assert menu_module.ModeManager.MODE_SANDFALL in manager.modes


def test_menu_has_no_duplicate_or_self_entries(monkeypatch):
    menu_module = _load_menu_module(monkeypatch)
    manager = DummyModeManager()
    menu = menu_module.Menu(width=28, height=28, mode_manager=manager)

    labels = [item.label for page in menu.pages for item in page]
    assert len(labels) == len(set(labels))
    assert "MENU" not in labels


def test_controller_navigation_suppresses_pointer_dwell(monkeypatch):
    menu_module = _load_menu_module(monkeypatch)
    now = {"value": 10.0}
    monkeypatch.setattr(menu_module.time, "time", lambda: now["value"])

    manager = DummyModeManager()
    menu = menu_module.Menu(width=28, height=28, mode_manager=manager)

    # Point at row 0 (CLOCK) and keep pointer steady long enough that normal
    # dwell behavior would click it.
    pointer = types.SimpleNamespace(source="web", x=0.10, y=0.05)
    hub = DummyInputHub(pointer=pointer, clicks=[])

    menu.mark_controller_navigation_active(now=now["value"])
    menu.get_frame(pose_results=None, input_hub=hub)

    now["value"] = now["value"] + 2.1
    menu.get_frame(pose_results=None, input_hub=hub)

    # No delayed click should occur while controller navigation suppression is active.
    assert manager.modes == []


def test_menu_ignores_controller_click_when_gesture_is_active_source(monkeypatch):
    menu_module = _load_menu_module(monkeypatch)
    monkeypatch.setattr(menu_module.time, "time", lambda: 1.0)

    manager = DummyModeManager()
    manager.allowed_sources = {"pose", "web"}
    menu = menu_module.Menu(width=28, height=28, mode_manager=manager)
    click = types.SimpleNamespace(source="controller", x=0.10, y=0.05)
    hub = DummyInputHub(pointer=None, clicks=[click])

    menu.get_frame(pose_results=None, input_hub=hub)

    assert manager.modes == []


def test_checkbox_provider_syncs_with_external_state(monkeypatch):
    menu_module = _load_menu_module(monkeypatch)
    state = {"on": True}
    checkbox = menu_module.Checkbox(
        "POSE", 0, 28, checked=True, checked_provider=lambda: state["on"]
    )
    frame = np.zeros((28, 28), dtype=np.uint8)

    # The setting flips elsewhere (e.g. via the web UI); drawing re-syncs.
    state["on"] = False
    checkbox.draw(frame)

    assert checkbox.checked is False


def test_checkbox_click_accepts_a_source(monkeypatch):
    # Every activation path (hover dwell, controller, web) passes a source;
    # Checkbox.click must accept it like every other MenuItem.
    menu_module = _load_menu_module(monkeypatch)
    calls = {"count": 0}
    checkbox = menu_module.Checkbox(
        "POSE",
        0,
        28,
        checked=False,
        on_click=lambda source=None: calls.__setitem__("count", calls["count"] + 1),
    )

    checkbox.click("web")
    checkbox.click("controller")
    checkbox.click()

    assert checkbox.checked is True  # toggled three times from False
    assert calls["count"] == 3


def test_checkbox_hover_dwell_click_does_not_crash(monkeypatch):
    menu_module = _load_menu_module(monkeypatch)
    now = {"value": 10.0}
    monkeypatch.setattr(menu_module.time, "time", lambda: now["value"])
    calls = {"count": 0}
    checkbox = menu_module.Checkbox(
        "POSE",
        0,
        28,
        checked=False,
        on_click=lambda source=None: calls.__setitem__("count", calls["count"] + 1),
    )

    checkbox.hover(True, source="gesture")
    now["value"] += menu_module.MenuItem.CLICK_TIME + 0.1
    checkbox.hover(True, source="gesture")

    assert calls["count"] == 1
    assert checkbox.checked is True
