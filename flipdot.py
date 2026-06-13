import numpy as np
import time
from app.services.fps import FPSTracker
from app.infrastructure.camera import Camera
from app.infrastructure.panel import Panel
import app.services.human_pose as human_pose
import app.services.image as image
import app.services.text as text
from app.core.mode_manager import ModeManager
from app.core.transition_policy import TransitionPolicy
from app.modes.contracts import RenderContext
from app.modes.registry import build_mode_registry
from app.modes.factory import create_mode_instances
from dotenv import load_dotenv
import os

PRINT_INTERVAL = 1.0
POSE_TIMEOUT = 2.0
CLOCK_RESOLVE_TIME = 1.0
CLOCK_DISOLVE_TIME = 1.0
SPIN_WAIT_MIN_FPS = 20
SPIN_GUARD_SEC = 0.012

load_dotenv()
print("Configuration:")
CAMERA_INDEX = int(os.getenv('CAMERA_INDEX', 0))
PREVIEW = os.getenv('PREVIEW', 'false').lower() == 'true'
DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
SLEEP_HOUR_START = int(os.getenv('SLEEP_HOUR_START', '0'))
SLEEP_HOUR_END = int(os.getenv('SLEEP_HOUR_END', '7'))

cam = Camera(CAMERA_INDEX)
panel = Panel(preview=PREVIEW)
fps_tracker = FPSTracker()

last_print_time = time.time()
last_update_time = time.time()
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
    img_sleep=img_sleep,
    clock_resolve_time=CLOCK_RESOLVE_TIME,
    clock_disolve_time=CLOCK_DISOLVE_TIME,
)
transition_policy = TransitionPolicy(
    pose_timeout=POSE_TIMEOUT,
    sleep_start_hour=SLEEP_HOUR_START,
    sleep_end_hour=SLEEP_HOUR_END,
)

while True:
    t_start = time.time()
    frame = cam.read_frame()
    frame = image.crop(frame)

    capture_time = time.time() - t_start

    dots = np.zeros((panel.HEIGHT,panel.WIDTH), dtype=np.uint8)

    t_process_start = time.time()

    if transition_policy.is_sleep_hour():
        pose_results = None
    else:
        pose_results = human_pose.get_human_pose(frame)
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
    )
    dots = mode_registry.render(mode_manager.mode, render_context)

    process_time = time.time() - t_process_start
    fps_limit = mode_manager.get_fps_limit()
    # Run at full speed in clock mode when a body is in frame, so the
    # transition to POSE mode feels immediate.
    body_in_frame = pose_results is not None and pose_results.pose_landmarks is not None
    if mode_manager.mode == ModeManager.MODE_CLOCK and body_in_frame:
        fps_limit = 30

    if DEBUG:
        dots[22:, :] = 0  # Clear bottom part of the panel
        dots[-1,-1] = fps_tracker.total_frames % 2
        estimated_distance = transition_state.estimated_distance
        angle = transition_state.angle
        estimated_distance_str = f"{estimated_distance:.1f}" if estimated_distance is not None else " "
        angle_str = f"{angle:02.0f}°" if angle is not None else " "
        text.write(dots, f"{estimated_distance_str}  {angle_str}", y=23, size=5)

    t_panel_start = time.time()
    panel.update(dots)
    panel_time = time.time() - t_panel_start

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

    last_update_time = time.time()
    fps_tracker.add_frame(capture_time, process_time, panel_time, sleep_time)
    current_time = time.time()
    if current_time - last_print_time >= PRINT_INTERVAL:
        stats = fps_tracker.get_stats()
        estimated_distance = transition_state.estimated_distance
        estimated_distance_str = f"{estimated_distance:.1f}" if estimated_distance is not None else "None"
        print(f"\rMode: {mode_manager.mode} | Eyes: {transition_state.eyes_visible} {transition_state.reason} | Dist: {estimated_distance_str} | FPS: {stats['fps']:.1f} (avg {stats['avg_fps']:.1f}) | Cap: {stats['capture_ms']:.0f}ms Proc: {stats['process_ms']:.0f}ms Panel: {stats['panel_ms']:.0f}ms Sleep: {stats['sleep_ms']:.0f}ms Total: {stats['total_ms']:.0f}ms | ", end='', flush=True)
        last_print_time = current_time
