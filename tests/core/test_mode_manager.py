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


def test_set_mode_pose_honored_even_when_pose_disabled(monkeypatch):
    # pose_enabled only governs the auto sandfall/caricature chain; explicit
    # POSE requests (MCP, menu fallback) must be honored, not redirected.
    fake = FakeClock(start=10.0)
    monkeypatch.setattr(mode_manager_module.time, "time", fake.time)

    manager = mode_manager_module.ModeManager(mode=mode_manager_module.ModeManager.MODE_CLOCK)
    manager.pose_enabled = False

    fake.now = 12.0
    manager.set_mode(mode_manager_module.ModeManager.MODE_POSE)

    assert manager.mode == mode_manager_module.ModeManager.MODE_POSE


def test_set_pose_enabled_notifies_change_hook():
    manager = mode_manager_module.ModeManager(mode=mode_manager_module.ModeManager.MODE_CLOCK)
    seen = []
    manager.on_pose_enabled_changed = seen.append

    manager.set_pose_enabled(False)
    manager.toggle_pose_enabled()

    assert seen == [False, True]


def test_set_pose_enabled_survives_failing_hook_and_records_it():
    manager = mode_manager_module.ModeManager(mode=mode_manager_module.ModeManager.MODE_CLOCK)

    def broken_hook(enabled):
        raise OSError("disk full")

    manager.on_pose_enabled_changed = broken_hook
    manager.set_pose_enabled(False)

    assert manager.pose_enabled is False
    # The live toggle applied but persistence failed; the web API reads this
    # to report a 500 instead of a silent success.
    assert manager.pose_persist_failed is True


def test_set_pose_enabled_noop_does_not_fire_hook():
    # Redundant web/MCP requests must not rewrite the settings file.
    manager = mode_manager_module.ModeManager(mode=mode_manager_module.ModeManager.MODE_CLOCK)
    seen = []
    manager.on_pose_enabled_changed = seen.append

    manager.set_pose_enabled(True)  # already enabled

    assert seen == []


def test_set_pose_enabled_noop_retries_failed_persist():
    # A same-value set after a failed persist must retry the write, or the
    # stale persisted value silently survives until the next real toggle.
    manager = mode_manager_module.ModeManager(mode=mode_manager_module.ModeManager.MODE_CLOCK)
    calls = {"count": 0, "fail": True}

    def flaky_hook(enabled):
        calls["count"] += 1
        if calls["fail"]:
            raise OSError("disk full")

    manager.on_pose_enabled_changed = flaky_hook
    manager.set_pose_enabled(False)
    assert manager.pose_persist_failed is True

    calls["fail"] = False
    manager.set_pose_enabled(False)  # same value: retried because it failed

    assert calls["count"] == 2
    assert manager.pose_persist_failed is False

    manager.set_pose_enabled(False)  # persisted fine: back to a true no-op
    assert calls["count"] == 2


def test_toggle_menu_close_marks_restore_and_fresh_set_mode_clears_it():
    manager = mode_manager_module.ModeManager(mode=mode_manager_module.ModeManager.MODE_SANDFALL)

    manager.set_mode(mode_manager_module.ModeManager.MODE_MENU)
    assert manager.consume_menu_restore() is False

    manager.toggle_menu()
    assert manager.mode == mode_manager_module.ModeManager.MODE_SANDFALL
    assert manager.consume_menu_restore() is True
    # One-shot: consuming clears the flag.
    assert manager.consume_menu_restore() is False

    manager.toggle_menu()
    manager.set_mode(mode_manager_module.ModeManager.MODE_TETRIS)
    assert manager.consume_menu_restore() is False


def test_retarget_menu_restore_redirects_close_target():
    manager = mode_manager_module.ModeManager(mode=mode_manager_module.ModeManager.MODE_SANDFALL)
    manager.set_mode(mode_manager_module.ModeManager.MODE_MENU)

    manager.retarget_menu_restore(mode_manager_module.ModeManager.MODE_CLOCK)
    manager.toggle_menu()

    assert manager.mode == mode_manager_module.ModeManager.MODE_CLOCK


def test_retarget_menu_restore_is_noop_outside_menu():
    manager = mode_manager_module.ModeManager(mode=mode_manager_module.ModeManager.MODE_SANDFALL)
    manager.set_mode(mode_manager_module.ModeManager.MODE_TETRIS)

    manager.retarget_menu_restore(mode_manager_module.ModeManager.MODE_CLOCK)

    assert manager.last_mode == mode_manager_module.ModeManager.MODE_SANDFALL


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
    assert (
        manager.get_fps_limit()
        == mode_manager_module.ModeManager.MAX_FPS[mode_manager_module.ModeManager.MODE_CLOCK]
    )


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
