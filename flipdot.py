import numpy as np
import time
from datetime import datetime
from fps import FPSTracker
from camera import Camera
from panel import Panel
from weather import get_weather_forecast
import human_pose
import transition
import image
from mode_manager import ModeManager
from menu import Menu
from clock import Clock
from paint import Paint
from caricature import Caricature
from percussion import Percussion
from autodrum import AutoDrum
from dotenv import load_dotenv
import os
import text

PRINT_INTERVAL = 1.0
POSE_TIMEOUT = 2.0
CLOCK_RESOLVE_TIME = 1.0
CLOCK_DISOLVE_TIME = 1.0

load_dotenv()
print("Configuration:")
CAMERA_INDEX = int(os.getenv('CAMERA_INDEX', 0))
PREVIEW = os.getenv('PREVIEW', 'false').lower() == 'true'
DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'

cam = Camera(CAMERA_INDEX)
panel = Panel(preview=PREVIEW)
fps_tracker = FPSTracker()
clock = Clock(panel.WIDTH, panel.HEIGHT)

last_print_time = time.time()
last_update_time = time.time()
pose_throttle_frame = 0
cached_drum_pose = None
mode_manager = ModeManager()
menu = Menu(panel.WIDTH, panel.HEIGHT, mode_manager)
paint = Paint(panel.WIDTH, panel.HEIGHT, mode_manager)
caricature = Caricature(panel.WIDTH, panel.HEIGHT, mode_manager)
percussion = Percussion(panel.WIDTH, panel.HEIGHT, mode_manager)
autodrum = AutoDrum(panel.WIDTH, panel.HEIGHT, mode_manager)
img_sleep = image.load('sleep.png')

while True:
    t_start = time.time()
    frame = cam.read_frame()
    frame = image.crop(frame)

    capture_time = time.time() - t_start

    dots = np.zeros((panel.HEIGHT,panel.WIDTH), dtype=np.uint8)

    t_process_start = time.time()

    pose_results = None
    face_mesh_results = None
    eyes_visible = False
    reason = ""
    now = datetime.now()
    estimated_distance = None
    angle = None
    if now.hour < 7 or now.hour >= 24:
        mode_manager.set_mode(ModeManager.MODE_SLEEP)
    else:
        current_mode = mode_manager.mode

        if current_mode == ModeManager.MODE_MENU:
            pose_results = human_pose.get_human_pose(frame)

        elif current_mode == ModeManager.MODE_PAINT:
            pose_results = human_pose.get_human_pose(frame)
            if human_pose.is_arms_crossed(pose_results):
                mode_manager.click_menu()
            else:
                mode_manager.reset_menu_click()

        elif current_mode == ModeManager.MODE_CARICATURE:
            pass  # no pose processing needed; caricature handles its own state

        elif current_mode in (ModeManager.MODE_PERCUSSION, ModeManager.MODE_AUTODRUM):
            # Throttle pose to every 6th frame (~5 FPS): interaction latency
            # is fine at 200 ms; sequencer timing is wall-clock independent.
            pose_throttle_frame = (pose_throttle_frame + 1) % 6
            if pose_throttle_frame == 0:
                cached_drum_pose = human_pose.get_human_pose(frame)
            pose_results = cached_drum_pose
            if human_pose.is_arms_crossed(pose_results):
                mode_manager.click_menu()
            else:
                mode_manager.reset_menu_click()

        elif current_mode == ModeManager.MODE_POSE:
            pose_results = human_pose.get_human_pose(frame)
            eyes_visible, reason, angle = human_pose.eyes_visible_and_facing_camera(pose_results)
            estimated_distance, _ = human_pose.estimate_distance(pose_results)
            if human_pose.should_draw_face_features(estimated_distance):
                face_mesh_results = human_pose.get_face_mesh(frame)

            if pose_results.pose_landmarks:
                mode_manager.set_mode(ModeManager.MODE_POSE)
            elif mode_manager.get_time_since_last_mode_update() > POSE_TIMEOUT:
                mode_manager.set_mode(ModeManager.MODE_CLOCK)

            if human_pose.is_arms_crossed(pose_results) and eyes_visible:
                mode_manager.click_menu()
            else:
                mode_manager.reset_menu_click()

        else:  # Clock and fallback modes
            if mode_manager.pose_enabled:
                pose_results = human_pose.get_human_pose(frame)
                eyes_visible, reason, angle = human_pose.eyes_visible_and_facing_camera(pose_results)
                estimated_distance, _ = human_pose.estimate_distance(pose_results)
                if pose_results.pose_landmarks and eyes_visible and estimated_distance < 1.3:
                    if mode_manager.mode not in (ModeManager.MODE_MENU, ModeManager.MODE_PAINT, ModeManager.MODE_CARICATURE):
                        mode_manager.set_mode(ModeManager.MODE_POSE)

            if human_pose.is_arms_crossed(pose_results) and eyes_visible:
                mode_manager.click_menu()
            else:
                mode_manager.reset_menu_click()

    if mode_manager.mode == ModeManager.MODE_SLEEP:
        dots[:,:] = img_sleep
    elif mode_manager.mode == ModeManager.MODE_POSE:
        dots = human_pose.display_human_pose(
            pose_results,
            panel.WIDTH,
            panel.HEIGHT,
            estimated_distance,
            face_mesh_results,
        )
        if mode_manager.get_mode_time() < CLOCK_DISOLVE_TIME:
            clock_dots = clock.get_frame()
            dots = transition.blend(clock_dots, dots, mode_manager.get_mode_time()/CLOCK_DISOLVE_TIME)
    elif mode_manager.mode == ModeManager.MODE_MENU:
        dots = menu.get_frame(pose_results)
    elif mode_manager.mode == ModeManager.MODE_CLOCK:
        dots = clock.get_frame()
        if not mode_manager.pose_enabled:
            human_pose.draw_right_index_pointer(dots, pose_results, size=2)
        if mode_manager.get_mode_time() < CLOCK_RESOLVE_TIME:
            dots = transition.resolve(dots, (mode_manager.get_mode_time())/CLOCK_RESOLVE_TIME)
    elif mode_manager.mode == ModeManager.MODE_PAINT:
        dots = paint.get_frame(pose_results)
    elif mode_manager.mode == ModeManager.MODE_CARICATURE:
        dots = caricature.get_frame(frame)
    elif mode_manager.mode == ModeManager.MODE_PERCUSSION:
        dots = percussion.get_frame(pose_results)
    elif mode_manager.mode == ModeManager.MODE_AUTODRUM:
        dots = autodrum.get_frame(pose_results)

    process_time = time.time() - t_process_start
    fps_limit = mode_manager.get_fps_limit()

    process_time = time.time() - t_process_start
    fps_limit = mode_manager.get_fps_limit()

    if DEBUG:
        dots[22:, :] = 0  # Clear bottom part of the panel
        dots[-1,-1] = fps_tracker.total_frames % 2
        estimated_distance_str = f"{estimated_distance:.1f}" if estimated_distance is not None else " "
        angle_str = f"{angle:02.0f}°" if angle is not None else " "
        text.write(dots, f"{estimated_distance_str}  {angle_str}", y=23, size=5)

    t_panel_start = time.time()
    panel.update(dots)
    panel_time = time.time() - t_panel_start

    # Precision frame limiter: sleep most of the budget then spin the last
    # ~6 ms so OS scheduling jitter doesn't overshoot the target.
    target_time = t_start + 1.0 / fps_limit
    remaining = target_time - time.time()
    t_sleep_start = time.time()
    if remaining > 0.006:
        time.sleep(remaining - 0.006)
    while time.time() < target_time:
        pass
    sleep_time = time.time() - t_sleep_start

    last_update_time = time.time()
    fps_tracker.add_frame(capture_time, process_time, panel_time, sleep_time)
    current_time = time.time()
    if current_time - last_print_time >= PRINT_INTERVAL:
        stats = fps_tracker.get_stats()
        estimated_distance_str = f"{estimated_distance:.1f}" if estimated_distance is not None else "None"
        print(f"\rMode: {mode_manager.mode} | Eyes: {eyes_visible} {reason} | Dist: {estimated_distance_str} | FPS: {stats['fps']:.1f} (avg {stats['avg_fps']:.1f}) | Cap: {stats['capture_ms']:.0f}ms Proc: {stats['process_ms']:.0f}ms Panel: {stats['panel_ms']:.0f}ms Sleep: {stats['sleep_ms']:.0f}ms Total: {stats['total_ms']:.0f}ms | ", end='', flush=True)
        last_print_time = current_time
