# AGENTS.md

Interactive art installation driving a **28×7 × 4-module flip-dot display** (28×28 effective, see `Panel`). A webcam feeds MediaPipe pose detection; the system switches between display "modes" based on whether a person is present, looking at the camera, and gesturing.

## Architecture & data flow

`flipdot.py` is the single entry point and main loop — there is no framework. Each iteration:
1. `Camera.read_frame()` → `image.crop()` to square → MediaPipe pose (`human_pose.get_human_pose`, downscaled to 60×60).
2. `ModeManager` decides the active mode from pose state (eyes visible + `estimate_distance < 1.3` → POSE; no pose for `POSE_TIMEOUT` → CLOCK; hours <7 or ≥24 → SLEEP).
3. The active mode renders a frame; `Panel.update(dots)` serializes and writes to flip-dot hardware.

**The frame is the universal data type:** a `np.zeros((HEIGHT, WIDTH), dtype=np.uint8)` array of 0/1 dot values. Every renderer (`Clock`, `Menu`, `Paint`, `human_pose`, `text`, `image`) takes and/or returns one. `Panel.WIDTH`/`Panel.HEIGHT` are the source of truth for dimensions — never hardcode them.

Modes (`ModeManager.MODE_*`): `sleep`, `clock`, `pose`, `menu`, `paint`. Each has a per-mode FPS cap in `ModeManager.MAX_FPS`; the loop sleeps to honor `get_fps_limit()` (always 30 for the first 5s of a mode for responsive transitions).

## Project-specific conventions

- **Drawing = direct numpy indexing.** Set/XOR pixels by slicing the frame: `frame[row*8:row*8+7, 0:slice] ^= 1`. Menu rows are 8px tall (`row*8`). See `menu.py` `Checkbox.draw` and `clock.py` `update_frame`.
- **Text** uses a hand-coded bitmap font in `text.py` (`FONTS[5]`, `FONTS[6]`). Render with `text.write(frame, str, x=, y=, size=5|6)`. Only glyphs present in `FONTS` exist — add new bitmaps there if needed.
- **Transitions** in `transition.py` operate on frames: `blend(a, b, alpha)`, `resolve`/`disolve(dots, alpha)` use random pixel masks (no greyscale — display is 1-bit).
- **Interaction is dwell-based**, not clicks: hover the right index finger over a `MenuItem` for `CLICK_TIME` (2s) to trigger `on_click`. `human_pose.is_arms_crossed` held for 2s opens the menu from pose, clock, and paint modes (via `mode_manager.click_menu()`/`reset_menu_click()` in the main loop).
- Mode classes that need to switch modes take `mode_manager` in their constructor and call `mode_manager.set_mode(...)` from `on_click` lambdas (see `Menu.__init__`).

## Hardware & environment

- **Device:** NVIDIA Jetson Orin Nano (aarch64), JetPack R36.4.7, running Ubuntu with Python 3.10.
- **Hardware deps:** real serial flip-dot panel at `/dev/ttyUSB0` (57600 baud) and a V4L2 webcam at `/dev/video0`.
- Run with `PREVIEW=true` to use `flippydot`'s on-screen pygame preview instead of serial — essential for dev without hardware.
- Config via `.env` (loaded by `python-dotenv`): `CAMERA_INDEX`, `PREVIEW`, `DEBUG`, `LOG_LEVEL`, `OPENWEATHER_API_KEY`. `DEBUG=true` overlays distance/angle text on the bottom rows; `LOG_LEVEL` controls Python logging verbosity (default `INFO`, set `DEBUG` for per-second performance logs).
- `flippydot/` is a vendored fork of the flip-dot driver library; `Panel` (`panel.py`) wraps it.

## Installed software on the Jetson

- **Python 3.10** (`/usr/bin/python3`) — packages installed system-wide via `pip3`, **not pipenv**.
- **mediapipe 0.10.18** — uses the Tasks API (`mediapipe.tasks`) with CPU/XNNPACK delegate. Model files live in `~/flipdot/models/` on the Jetson (excluded from rsync and `.gitignore`). If model files are absent (e.g. on a dev machine) the code automatically falls back to the legacy `mp.solutions.pose` API, so local development works without them. GPU delegate is not available in the pip build; TensorRT 10.3.0 is installed but not yet wired up.
- **TensorRT 10.3.0** — pre-installed with JetPack; future path for GPU-accelerated pose inference.
- **opencv-python, pyserial, requests, pillow, python-dotenv** — installed via pip3.

## SSH access

```bash
ssh flipdot          # connects as flipdot@flipdot (host alias in ~/.ssh/config)
```

Useful commands on the device:
```bash
sudo systemctl status flipdot.service   # check if running
sudo systemctl restart flipdot.service  # restart after deploy
sudo tail -f /var/log/flipdot/output.log   # app INFO/DEBUG logs
sudo tail -f /var/log/flipdot/error.log    # tracebacks and stderr
sudo logrotate -f /etc/logrotate.d/flipdot # force log rotation check
```

## Developer workflows

- **Run locally (dev machine):** `PREVIEW=true python3 flipdot.py`
- **Run tests locally:** `pipenv run pytest` (from repo root).
  - Current suite covers core state/policy modules, deterministic services, and selected mode/registry behavior under `tests/`.
  - Tests are designed to run without hardware/network and avoid hard dependency on MediaPipe model files.
- **Run on device:** `sudo systemctl start flipdot.service`; the service auto-restarts on crash (`Restart=always`).
- **Deploy:** `./deploy.sh [--debug]` — rsyncs to `flipdot@flipdot:/home/flipdot/flipdot` (with `--delete`, but `models/` and `.env` are excluded), sets `DEBUG` in `.env`, ensures `/var/log/flipdot` exists, installs `ops/systemd/flipdot.service` and `ops/logrotate/flipdot` on the device, then reloads and restarts `flipdot.service`.
  - **Important:** the `models/` directory is excluded from rsync. MediaPipe `.task` model files must be downloaded manually once:
    ```bash
    ssh flipdot
    mkdir -p ~/flipdot/models && cd ~/flipdot/models
    wget -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
    wget -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
    ```

## Gotchas

- `weather.py` reads OpenWeatherMap credentials from `.env`; missing `OPENWEATHER_API_KEY` returns an error payload instead of forecast data.
- Frames are 1-bit: anything you draw must end up as 0/1 `uint8`. Segmentation masks etc. must be thresholded (`(x > 0.5).astype(np.uint8)`).
- Pose results can be `None` — guard `pose_results.pose_landmarks` before use, as the main loop does.
