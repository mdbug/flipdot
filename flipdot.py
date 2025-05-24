import numpy as np
import time
from datetime import datetime
from fps import FPSTracker
from camera import Camera
from panel import Panel
from weather import get_weather_forecast
import human_pose
from transition import disolve, resolve
from image import load_image
from mode_manager import ModeManager
from clock import Clock
from dotenv import load_dotenv
import os

PRINT_INTERVAL = 1.0
POSE_TIMEOUT = 3.0
CLOCK_RESOLVE_TIME = 2.0
CLOCK_DISOLVE_TIME = 2.0

load_dotenv()
CAMERA_INDEX = int(os.getenv('CAMERA_INDEX', 0))
PREVIEW = os.getenv('PREVIEW', 'false').lower() == 'true'

cam = Camera(CAMERA_INDEX)
panel = Panel(preview=PREVIEW)
fps_tracker = FPSTracker()
clock = Clock(panel.WIDTH, panel.HEIGHT)

last_print_time = time.time()
last_update_time = time.time()
mode_manager = ModeManager()
img_sleep = load_image('sleep.png')

while True:
    t_start = time.time()
    frame = cam.read_frame()
    capture_time = time.time() - t_start

    dots = np.zeros((panel.HEIGHT,panel.WIDTH), dtype=np.uint8)

    t_process_start = time.time()

    pose_results = None
    eyes_visible = False
    now = datetime.now()
    if now.hour < 7 or now.hour >= 24:
        mode_manager.set_mode('sleep')
    else:
        pose_results = human_pose.get_human_pose(frame)
        # eyes_visible = human_pose.eyes_visible(pose_results)
        eyes_visible, reason = human_pose.check_eyes_visible_and_facing_camera(pose_results)
        print(reason)
        if pose_results.pose_landmarks:
            if mode_manager.mode == mode_manager.MODE_POSE:
                mode_manager.set_mode('pose')
            elif eyes_visible:
                mode_manager.set_mode('pose')
        else:
            if mode_manager.get_time_since_last_mode_update() > POSE_TIMEOUT:
                mode_manager.set_mode('clock')

    if mode_manager.mode == 'sleep':
        dots[:,:] = img_sleep
    elif mode_manager.mode == 'pose':
        dots = human_pose.display_human_pose(pose_results, panel.WIDTH, panel.HEIGHT)
        if mode_manager.get_mode_time() < CLOCK_DISOLVE_TIME:
            clock_dots = clock.get_frame()
            clock_dots = disolve(clock_dots, (mode_manager.get_mode_time())/CLOCK_DISOLVE_TIME)
            dots = np.logical_or(dots, clock_dots)
    elif mode_manager.mode == 'clock':
        dots = clock.get_frame()
        if mode_manager.get_mode_time() < CLOCK_RESOLVE_TIME:
            dots = resolve(dots, (mode_manager.get_mode_time())/CLOCK_RESOLVE_TIME)

    process_time = time.time() - t_process_start

    fps_limit = mode_manager.get_fps_limit()
    if (time.time() - last_update_time) < (1.0/fps_limit):
        time.sleep(1.0/fps_limit - (time.time() - last_update_time))

    panel.update(dots)
    last_update_time = time.time()
    fps_tracker.add_frame(capture_time, process_time)
    current_time = time.time()
    if current_time - last_print_time >= PRINT_INTERVAL:
        stats = fps_tracker.get_stats()
        print(f"Mode: {mode_manager.mode} | Eyes visible: {eyes_visible} | FPS: {stats['fps']:.1f} | Avg: {stats['avg_fps']:.1f} | ")
        last_print_time = current_time
