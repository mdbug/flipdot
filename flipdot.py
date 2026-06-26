import numpy as np
import time
import logging
from dotenv import load_dotenv
import os
from app.services.logging_setup import setup_logging

load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)

from app.services.fps import FPSTracker
from app.services.controller import ControllerHub
from app.services.controller_mapping import ControllerInputBridge
from app.infrastructure.camera import Camera
from app.infrastructure.panel import Panel
from app.core.input_source import InputHub
from app.core.action_dispatch import dispatch_actions
import app.services.human_pose as human_pose
import app.services.image as image
import app.services.text as text
from app.core.mode_manager import ModeManager
from app.core.transition_policy import TransitionPolicy
from app.modes.contracts import RenderContext
from app.modes.registry import build_mode_registry
from app.modes.factory import create_mode_instances

PRINT_INTERVAL = 1.0
POSE_TIMEOUT = 2.0
CLOCK_RESOLVE_TIME = 1.0
CLOCK_DISOLVE_TIME = 1.0
SPIN_WAIT_MIN_FPS = 20
SPIN_GUARD_SEC = 0.012
PAINT_CLEAR_HOLD_SEC = 1.0
PRIMARY_CONTROLLER_ADDRESS = os.getenv("PRIMARY_CONTROLLER_ADDRESS", "AA:BB:CC:DD:EE:02")
PRIMARY_CONTROLLER_NAME = os.getenv("PRIMARY_CONTROLLER_NAME", "IINE_keyboard")
SECONDARY_CONTROLLER_ADDRESS = os.getenv("SECONDARY_CONTROLLER_ADDRESS", "AA:BB:CC:DD:EE:03")

# Modes that are fully driven by the controller UI and never render pose.
# When the controller is the active control source, pose inference can be
# skipped for these modes so the render/input loop is not starved by the
# expensive per-frame MediaPipe call (keeping controller input responsive).
CONTROLLER_DRIVEN_UI_MODES = frozenset({
    ModeManager.MODE_MENU,
    ModeManager.MODE_PAINT,
    ModeManager.MODE_BOARD,
    ModeManager.MODE_TETRIS,
    ModeManager.MODE_PONG,
    ModeManager.MODE_PERCUSSION,
    ModeManager.MODE_AUTODRUM,
    ModeManager.MODE_FONT_PREVIEW,
})


def get_web_controls(mode: str) -> list[dict[str, str]]:
    controls: list[dict[str, str]] = [
        {
            "id": "toggle_menu",
            "label": "Menu",
            "variant": "accent",
            "action": "toggle_menu",
        }
    ]

    mode_controls: dict[str, list[dict[str, str]]] = {
        ModeManager.MODE_PAINT: [
            {
                "id": "paint_clear",
                "label": "Clear",
                "variant": "secondary",
                "action": "paint_clear",
            }
        ],
        ModeManager.MODE_AUTODRUM: [
            {
                "id": "autodrum_next_song",
                "label": "Next Song",
                "variant": "secondary",
                "action": "autodrum_next_song",
            }
        ],
        ModeManager.MODE_BOARD: [
            {
                "id": "board_clear",
                "label": "Clear",
                "variant": "secondary",
                "action": "board_clear",
            },
            {
                "id": "board_undo",
                "label": "Undo",
                "variant": "secondary",
                "action": "board_undo",
            },
        ],
        ModeManager.MODE_FONT_PREVIEW: [
            {
                "id": "font_prev",
                "label": "Prev",
                "variant": "secondary",
                "action": "font_preview_prev",
            },
            {
                "id": "font_next",
                "label": "Next",
                "variant": "secondary",
                "action": "font_preview_next",
            },
        ],
    }
    controls.extend(mode_controls.get(mode, []))
    return controls

def main():
    camera_index = int(os.getenv('CAMERA_INDEX', 0))
    preview = os.getenv('PREVIEW', 'false').lower() == 'true'
    debug = os.getenv('DEBUG', 'false').lower() == 'true'
    sleep_hour_start = int(os.getenv('SLEEP_HOUR_START', '0'))
    sleep_hour_end = int(os.getenv('SLEEP_HOUR_END', '7'))
    face_mesh_max_fps = float(os.getenv('FACE_MESH_MAX_FPS', '12'))
    enable_web_ui = os.getenv('ENABLE_WEB_UI', 'false').lower() == 'true'
    web_ui_host = os.getenv('WEB_UI_HOST', '0.0.0.0')
    web_ui_port = int(os.getenv('WEB_UI_PORT', '8000'))

    logger.info(
        "Starting flipdot app camera_index=%s preview=%s debug=%s sleep_window=%s-%s",
        camera_index,
        preview,
        debug,
        sleep_hour_start,
        sleep_hour_end,
    )

    cam = Camera(camera_index)
    panel = Panel(preview=preview)
    fps_tracker = FPSTracker()
    input_hub = InputHub()
    primary_controller_hub = ControllerHub(
        target_address=PRIMARY_CONTROLLER_ADDRESS,
        target_name_hint=PRIMARY_CONTROLLER_NAME,
    )
    secondary_controller_hub = ControllerHub(target_address=SECONDARY_CONTROLLER_ADDRESS)
    controller_bridge = ControllerInputBridge()
    web_server = None
    web_server_start_pending = enable_web_ui

    last_log_time = time.time()
    paint_clear_gesture_start = None
    paint_clear_gesture_armed = True
    mode_manager = ModeManager()
    mode_instances = create_mode_instances(panel.WIDTH, panel.HEIGHT, mode_manager)
    clock = mode_instances["clock"]
    menu = mode_instances["menu"]
    paint = mode_instances["paint"]
    caricature = mode_instances["caricature"]
    percussion = mode_instances["percussion"]
    autodrum = mode_instances["autodrum"]
    beatmirror = mode_instances["beatmirror"]
    tetris_game = mode_instances["tetris"]
    pong_game = mode_instances["pong"]
    worldcup = mode_instances["worldcup"]
    board = mode_instances["board"]
    font_preview = mode_instances["font_preview"]
    script_mode = mode_instances["script"]
    img_sleep = image.load('sleep.png')
    mode_registry = build_mode_registry(
        clock=clock,
        menu=menu,
        paint=paint,
        caricature=caricature,
        percussion=percussion,
        autodrum=autodrum,
        beatmirror=beatmirror,
        tetris_game=tetris_game,
        pong_game=pong_game,
        worldcup=worldcup,
        board=board,
        font_preview=font_preview,
        script_mode=script_mode,
        img_sleep=img_sleep,
        clock_resolve_time=CLOCK_RESOLVE_TIME,
        clock_disolve_time=CLOCK_DISOLVE_TIME,
    )
    transition_policy = TransitionPolicy(
        pose_timeout=POSE_TIMEOUT,
        sleep_start_hour=sleep_hour_start,
        sleep_end_hour=sleep_hour_end,
        face_mesh_max_fps=face_mesh_max_fps,
    )

    def get_controller_statuses() -> list[dict]:
        # Keep primary first so existing consumers can treat index 0 as the
        # first status while the gameplay bridge accepts either controller.
        return [
            primary_controller_hub.get_status_snapshot(),
            secondary_controller_hub.get_status_snapshot(),
        ]

    def merge_controller_snapshots(snapshots: list[dict]) -> dict:
        connected_snapshots = [
            snapshot
            for snapshot in snapshots
            if bool(snapshot.get("enabled")) and bool(snapshot.get("connected"))
        ]
        primary_snapshot = snapshots[0] if snapshots else {}
        source_snapshot = connected_snapshots[0] if connected_snapshots else primary_snapshot

        pressed_buttons = set()
        last_event_monotonic = None
        for snapshot in connected_snapshots:
            pressed_buttons.update(str(button) for button in snapshot.get("pressed_buttons", []))
            event_time = snapshot.get("last_event_monotonic")
            if isinstance(event_time, (int, float)) and (
                last_event_monotonic is None or event_time > last_event_monotonic
            ):
                last_event_monotonic = event_time

        return {
            "enabled": any(bool(snapshot.get("enabled")) for snapshot in snapshots),
            "connected": bool(connected_snapshots),
            "address": str(source_snapshot.get("address", "") or ""),
            "device_name": str(source_snapshot.get("device_name", "") or ""),
            "pressed_buttons": sorted(pressed_buttons),
            "last_event_monotonic": last_event_monotonic,
            "battery_percentage": source_snapshot.get("battery_percentage"),
            "battery_updated_monotonic": source_snapshot.get("battery_updated_monotonic"),
        }

    try:
        while True:
            t_start = time.time()
            frame = cam.read_frame()
            frame = image.crop(frame)

            capture_time = time.time() - t_start

            dots = np.zeros((panel.HEIGHT, panel.WIDTH), dtype=np.uint8)

            t_process_start = time.time()

            controller_snapshots = get_controller_statuses()
            controller_snapshot = merge_controller_snapshots(controller_snapshots)
            controller_active = bool(controller_snapshot.get("enabled")) and bool(controller_snapshot.get("connected"))
            mode_manager.update_controller_connected(controller_active)

            controller_pressed_events = (
                primary_controller_hub.drain_pressed_events()
                | secondary_controller_hub.drain_pressed_events()
            )
            controller_bridge.process(
                snapshot=controller_snapshot,
                primary_snapshot=controller_snapshots[0] if controller_snapshots else None,
                secondary_snapshot=controller_snapshots[1] if len(controller_snapshots) > 1 else None,
                mode=mode_manager.mode,
                input_hub=input_hub,
                mode_manager=mode_manager,
                menu=menu,
                paint=paint,
                autodrum=autodrum,
                board=board,
                font_preview=font_preview,
                tetris_game=tetris_game,
                pong_game=pong_game,
                percussion=percussion,
                pressed_events=controller_pressed_events,
            )

            # Skip the expensive per-frame pose inference when the controller is
            # driving a UI mode that does not render pose. This keeps the loop
            # fast and steady so controller input stays responsive (no MediaPipe
            # stall between input samples).
            controller_driving_ui = (
                mode_manager.get_effective_control_source() == ModeManager.CONTROL_CONTROLLER
                and mode_manager.mode in CONTROLLER_DRIVEN_UI_MODES
            )

            # Scripted animations never read pose data, so skip MediaPipe inference
            # entirely while they run — it's the main cost keeping the loop below
            # the 30 FPS target and gives the script a steady, load-independent tick.
            script_mode_active = mode_manager.mode == ModeManager.MODE_SCRIPT

            if transition_policy.is_sleep_hour() or controller_driving_ui or script_mode_active:
                pose_results = None
            else:
                pose_results = human_pose.get_human_pose(frame)
            input_hub.ingest_pose(pose_results)

            allowed_sources = mode_manager.get_allowed_input_sources(include_web=True)

            if mode_manager.mode == ModeManager.MODE_PAINT:
                if human_pose.is_left_hand_raised(pose_results):
                    if paint_clear_gesture_start is None:
                        paint_clear_gesture_start = time.time()
                    elif paint_clear_gesture_armed and (time.time() - paint_clear_gesture_start) >= PAINT_CLEAR_HOLD_SEC:
                        paint.clear()
                        paint_clear_gesture_armed = False
                else:
                    paint_clear_gesture_start = None
                    paint_clear_gesture_armed = True
            else:
                paint_clear_gesture_start = None
                paint_clear_gesture_armed = True

            dispatch_actions(
                actions=input_hub.pop_actions(allowed_sources=allowed_sources),
                mode_manager=mode_manager,
                paint=paint,
                autodrum=autodrum,
                board=board,
                font_preview=font_preview,
                allowed_sources=allowed_sources,
            )

            transition_state = transition_policy.apply(
                frame=frame,
                pose_results=pose_results,
                mode_manager=mode_manager,
                paint_mode=paint,
            )

            render_context = RenderContext(
                frame=frame,
                pose_results=pose_results,
                face_mesh_results=transition_state.face_mesh_results,
                estimated_distance=transition_state.estimated_distance,
                mode_time=mode_manager.get_mode_time(),
                panel_width=panel.WIDTH,
                panel_height=panel.HEIGHT,
                input_hub=input_hub,
            )
            dots = mode_registry.render(mode_manager.mode, render_context)

            process_time = time.time() - t_process_start
            fps_limit = mode_manager.get_fps_limit()
            if controller_active:
                fps_limit = max(fps_limit, 30)
            # Run at full speed in clock mode when a body is in frame, so the
            # transition to POSE mode feels immediate.
            body_in_frame = pose_results is not None and pose_results.pose_landmarks is not None
            if mode_manager.mode == ModeManager.MODE_CLOCK and body_in_frame:
                fps_limit = 30

            if debug:
                dots[22:, :] = 0  # Clear bottom part of the panel
                dots[-1, -1] = fps_tracker.total_frames % 2
                estimated_distance = transition_state.estimated_distance
                angle = transition_state.angle
                estimated_distance_str = f"{estimated_distance:.1f}" if estimated_distance is not None else " "
                angle_str = f"{angle:02.0f}°" if angle is not None else " "
                text.write(
                    dots,
                    f"{estimated_distance_str}  {angle_str}",
                    y=23,
                    size=5,
                    style="regular",
                )

            t_panel_start = time.time()
            panel.update(dots)
            panel_time = time.time() - t_panel_start
            panel_updated_monotonic = time.monotonic()

            if web_server_start_pending:
                # Defer FastAPI/uvicorn import/start until after the first panel
                # update so cold web stack startup never delays first pixels.
                from app.infrastructure.web_server import WebServer

                web_server = WebServer(input_hub=input_hub, host=web_ui_host, port=web_ui_port)
                web_server.attach_board(board)
                web_server.attach_script_mode(script_mode)
                web_server.attach_mode_manager(mode_manager)
                web_server.attach_font_preview(font_preview)
                web_server.attach_transition_policy(transition_policy)
                web_server.attach_controller_status_provider(get_controller_statuses)
                web_server.start()
                web_server_start_pending = False
                logger.info("Web UI enabled on http://%s:%s", web_ui_host, web_ui_port)

            if web_server is not None:
                web_server.publish_frame(
                    dots,
                    mode=mode_manager.mode,
                    controls=get_web_controls(mode_manager.mode),
                    panel_updated_monotonic=panel_updated_monotonic,
                )

            # Use precision spin-wait only for high-FPS modes; for low-FPS modes,
            # pure sleep yields CPU and reduces power usage.
            target_time = t_start + 1.0 / fps_limit
            remaining = target_time - time.time()
            t_sleep_start = time.time()
            if remaining > 0:
                if fps_limit >= SPIN_WAIT_MIN_FPS:
                    if remaining > SPIN_GUARD_SEC:
                        time.sleep(remaining - SPIN_GUARD_SEC)
                    while time.time() < target_time:
                        pass
                else:
                    time.sleep(remaining)
            sleep_time = time.time() - t_sleep_start

            fps_tracker.add_frame(capture_time, process_time, panel_time, sleep_time)
            current_time = time.time()
            if current_time - last_log_time >= PRINT_INTERVAL:
                stats = fps_tracker.get_stats()
                estimated_distance = transition_state.estimated_distance
                estimated_distance_str = f"{estimated_distance:.1f}" if estimated_distance is not None else "None"
                logger.debug(
                    "mode=%s eyes=%s reason=%s dist=%s fps=%.1f avg_fps=%.1f cap_ms=%.0f proc_ms=%.0f panel_ms=%.0f sleep_ms=%.0f total_ms=%.0f",
                    mode_manager.mode,
                    transition_state.eyes_visible,
                    transition_state.reason,
                    estimated_distance_str,
                    stats['fps'],
                    stats['avg_fps'],
                    stats['capture_ms'],
                    stats['process_ms'],
                    stats['panel_ms'],
                    stats['sleep_ms'],
                    stats['total_ms'],
                )
                last_log_time = current_time
    finally:
        if web_server is not None:
            web_server.stop()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Flipdot app terminated due to fatal error")
        raise
