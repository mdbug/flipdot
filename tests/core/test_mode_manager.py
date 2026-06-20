import app.core.mode_manager as mode_manager_module


class FakeClock:
    def __init__(self, start=0.0):
        self.now = float(start)

    def time(self):
        return self.now


def test_set_mode_updates_last_mode_and_timestamps(monkeypatch):
    fake = FakeClock(start=100.0)
    monkeypatch.setattr(mode_manager_module.time, "time", fake.time)

    manager = mode_manager_module.ModeManager(mode=mode_manager_module.ModeManager.MODE_CLOCK)
    assert manager.mode == mode_manager_module.ModeManager.MODE_CLOCK

    fake.now = 120.0
    manager.set_mode(mode_manager_module.ModeManager.MODE_MENU)

    assert manager.last_mode == mode_manager_module.ModeManager.MODE_CLOCK
    assert manager.mode == mode_manager_module.ModeManager.MODE_MENU
    assert manager.mode_start_time == 120.0
    assert manager.mode_update_time == 120.0


def test_set_mode_pose_falls_back_when_pose_disabled(monkeypatch):
    fake = FakeClock(start=10.0)
    monkeypatch.setattr(mode_manager_module.time, "time", fake.time)

    manager = mode_manager_module.ModeManager(mode=mode_manager_module.ModeManager.MODE_CLOCK)
    manager.pose_enabled = False

    fake.now = 12.0
    manager.set_mode(mode_manager_module.ModeManager.MODE_POSE)

    assert manager.mode == mode_manager_module.ModeManager.MODE_CLOCK


def test_click_menu_uses_dwell_to_toggle_menu(monkeypatch):
    fake = FakeClock(start=1.0)
    monkeypatch.setattr(mode_manager_module.time, "time", fake.time)

    manager = mode_manager_module.ModeManager(mode=mode_manager_module.ModeManager.MODE_CLOCK)

    manager.click_menu()
    assert manager.menu_click_start == 1.0
    assert manager.mode == mode_manager_module.ModeManager.MODE_CLOCK

    fake.now = 3.2
    manager.click_menu()
    assert manager.mode == mode_manager_module.ModeManager.MODE_MENU
    assert manager.menu_click_start is None

    manager.click_menu()
    fake.now = 5.4
    manager.click_menu()
    assert manager.mode == mode_manager_module.ModeManager.MODE_CLOCK


def test_toggle_menu_returns_to_previous_mode(monkeypatch):
    fake = FakeClock(start=30.0)
    monkeypatch.setattr(mode_manager_module.time, "time", fake.time)

    manager = mode_manager_module.ModeManager(mode=mode_manager_module.ModeManager.MODE_CLOCK)
    manager.toggle_menu()
    assert manager.mode == mode_manager_module.ModeManager.MODE_MENU

    fake.now = 31.0
    manager.toggle_menu()
    assert manager.mode == mode_manager_module.ModeManager.MODE_CLOCK


def test_get_fps_limit_warmup_then_mode_specific(monkeypatch):
    fake = FakeClock(start=50.0)
    monkeypatch.setattr(mode_manager_module.time, "time", fake.time)

    manager = mode_manager_module.ModeManager(mode=mode_manager_module.ModeManager.MODE_CLOCK)

    fake.now = 54.0
    assert manager.get_fps_limit() == 30

    fake.now = 56.0
    assert manager.get_fps_limit() == mode_manager_module.ModeManager.MAX_FPS[mode_manager_module.ModeManager.MODE_CLOCK]


def test_set_mode_updates_control_source_when_entered_via(monkeypatch):
    fake = FakeClock(start=20.0)
    monkeypatch.setattr(mode_manager_module.time, "time", fake.time)

    manager = mode_manager_module.ModeManager(mode=mode_manager_module.ModeManager.MODE_CLOCK)
    manager.update_controller_connected(True)
    manager.set_mode(
        mode_manager_module.ModeManager.MODE_MENU,
        entered_via=mode_manager_module.ModeManager.CONTROL_GESTURE,
    )

    assert manager.control_source == mode_manager_module.ModeManager.CONTROL_GESTURE


def test_set_mode_keeps_control_source_when_entered_via_missing(monkeypatch):
    fake = FakeClock(start=30.0)
    monkeypatch.setattr(mode_manager_module.time, "time", fake.time)

    manager = mode_manager_module.ModeManager(mode=mode_manager_module.ModeManager.MODE_CLOCK)
    manager.update_controller_connected(True)
    manager.set_mode(
        mode_manager_module.ModeManager.MODE_MENU,
        entered_via=mode_manager_module.ModeManager.CONTROL_CONTROLLER,
    )
    manager.set_mode(mode_manager_module.ModeManager.MODE_TETRIS)

    assert manager.control_source == mode_manager_module.ModeManager.CONTROL_CONTROLLER


def test_effective_control_source_falls_back_to_gesture_when_disconnected(monkeypatch):
    fake = FakeClock(start=40.0)
    monkeypatch.setattr(mode_manager_module.time, "time", fake.time)

    manager = mode_manager_module.ModeManager(mode=mode_manager_module.ModeManager.MODE_CLOCK)
    manager.control_source = mode_manager_module.ModeManager.CONTROL_CONTROLLER
    manager.update_controller_connected(False)

    assert manager.get_effective_control_source() == mode_manager_module.ModeManager.CONTROL_GESTURE


def test_controller_connect_immediately_switches_control_source(monkeypatch):
    fake = FakeClock(start=50.0)
    monkeypatch.setattr(mode_manager_module.time, "time", fake.time)

    manager = mode_manager_module.ModeManager(mode=mode_manager_module.ModeManager.MODE_MENU)
    manager.control_source = mode_manager_module.ModeManager.CONTROL_GESTURE

    switched = manager.update_controller_connected(True)

    assert switched is True
    assert manager.control_source == mode_manager_module.ModeManager.CONTROL_CONTROLLER
