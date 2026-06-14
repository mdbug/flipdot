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
from app.infrastructure.camera import Camera
from app.infrastructure.panel import Panel
from app.core.input_source import InputHub
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

    try:
        while True:
            t_start = time.time()
            frame = cam.read_frame()
            frame = image.crop(frame)

            capture_time = time.time() - t_start

            dots = np.zeros((panel.HEIGHT, panel.WIDTH), dtype=np.uint8)

            t_process_start = time.time()

            if transition_policy.is_sleep_hour():
                pose_results = None
            else:
                pose_results = human_pose.get_human_pose(frame)
            input_hub.ingest_pose(pose_results)

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

            for action in input_hub.pop_actions():
                if action.action == 'toggle_menu':
                    mode_manager.toggle_menu()
                elif action.action == 'paint_clear' and mode_manager.mode == ModeManager.MODE_PAINT:
                    paint.clear()
                elif action.action == 'autodrum_next_song' and mode_manager.mode == ModeManager.MODE_AUTODRUM:
                    autodrum.next_song()
                elif action.action == 'board_clear' and mode_manager.mode == ModeManager.MODE_BOARD:
                    board.clear()
                elif action.action == 'board_undo' and mode_manager.mode == ModeManager.MODE_BOARD:
                    board.undo()
                elif action.action == 'font_preview_prev' and mode_manager.mode == ModeManager.MODE_FONT_PREVIEW:
                    font_preview.previous_variant()
                elif action.action == 'font_preview_next' and mode_manager.mode == ModeManager.MODE_FONT_PREVIEW:
                    font_preview.next_variant()

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

            if web_server_start_pending:
                # Defer FastAPI/uvicorn import/start until after the first panel
                # update so cold web stack startup never delays first pixels.
                from app.infrastructure.web_server import WebServer

                web_server = WebServer(input_hub=input_hub, host=web_ui_host, port=web_ui_port)
                web_server.attach_board(board)
                web_server.attach_font_preview(font_preview)
                web_server.attach_transition_policy(transition_policy)
                web_server.start()
                web_server_start_pending = False
                logger.info("Web UI enabled on http://%s:%s", web_ui_host, web_ui_port)

            if web_server is not None:
                web_server.publish_frame(
                    dots,
                    mode=mode_manager.mode,
                    controls=get_web_controls(mode_manager.mode),
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
