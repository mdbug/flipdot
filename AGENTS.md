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
- **Interaction is dwell-based**, not clicks: hover the right index finger over a `MenuItem` for `CLICK_TIME` (2s) to trigger `on_click`. `human_pose.is_right_index_in_top_right_corner` opens the menu after a 2s hold.
- Mode classes that need to switch modes take `mode_manager` in their constructor and call `mode_manager.set_mode(...)` from `on_click` lambdas (see `Menu.__init__`).

## Hardware & environment

- **Hardware deps:** real serial flip-dot panel at `/dev/ttyUSB0` (57600 baud) and a V4L2 webcam. Run with `PREVIEW=true` to use `flippydot`'s on-screen pygame preview instead of serial — essential for dev without hardware.
- Config via `.env` (loaded by `python-dotenv`): `CAMERA_INDEX`, `PREVIEW`, `DEBUG`, `OPENWEATHER_API_KEY`. `DEBUG=true` overlays distance/angle text on the bottom rows.
- `flippydot/` is a vendored fork of the flip-dot driver library; `Panel` (`panel.py`) wraps it.

## Developer workflows

- **Run:** `pipenv install && pipenv run python flipdot.py` (Python 3.11; runtime deps include `opencv-python`, `mediapipe`, `requests`, `pyserial`, `pillow`, `python-dotenv`).
- **Deploy:** `./deploy.sh [--debug]` — rsyncs to the `flipdot@flipdot` host, sets `DEBUG`, and restarts `flipdot.service` (systemd). The device runs the loop as a service.
- **No test suite** for the app; only `flipPyDot/test/test.py` covers the vendored driver.

## Gotchas

- `weather.py` reads OpenWeatherMap credentials from `.env`; missing `OPENWEATHER_API_KEY` returns an error payload instead of forecast data.
- Frames are 1-bit: anything you draw must end up as 0/1 `uint8`. Segmentation masks etc. must be thresholded (`(x > 0.5).astype(np.uint8)`).
- Pose results can be `None` — guard `pose_results.pose_landmarks` before use, as the main loop does.
