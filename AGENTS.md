# AGENTS.md

Interactive art installation driving a **28×7 × 4-module flip-dot display** (28×28 effective, see `Panel`). A webcam feeds MediaPipe pose detection; the system switches between display "modes" based on whether a person is present, looking at the camera, and gesturing. The display can additionally be driven by Bluetooth game controllers, a browser web UI, and an AI agent (in-UI Claude chat or an external MCP client).

## Code layout

`flipdot.py` at the repo root is the single entry point and main loop. Everything else lives under the `app/` package:

- `app/core/` — the loop's decision logic, hardware-free and unit-tested: `ModeManager` (active mode + control source), `TransitionPolicy` (decides mode from pose/time), `InputHub` (`input_source.py`, merges input from all sources into a queue of pointer/click/action events), `action_dispatch` (applies queued actions to modes).
- `app/modes/` — one class per display mode (see list below) plus the renderer plumbing: `contracts.py` (`Frame`, `RenderContext`, `ModeRegistry`), `factory.py` (`create_mode_instances`), `registry.py` (`build_mode_registry` maps a mode id to a renderer callable).
- `app/infrastructure/` — I/O boundaries: `Camera`, `Panel` (flip-dot hardware), `WebServer` (FastAPI), `mcp_server.py` (AI tools), `chat.py` (Claude backend).
- `app/services/` — supporting logic: `human_pose`, `text`, `draw` (1-bit line/circle/point primitives), `image`, `transition`, `weather`, `worldcup`, `fps`, `controller`/`controller_mapping` (BLE gamepads), `sandbox`/`script_store` (scripted animations), `settings_store`/`chat_session_store` (persistence), `fonts/`.
- `web_ui/` — the browser console (static HTML/JS/CSS served by `WebServer`).
- `state/` — runtime persistence (settings, saved boards, scripts, chat sessions); excluded from deploy rsync.
- `flipPyDot/` — vendored fork of the flip-dot driver library (installed as the `flippydot` package); `Panel` (`app/infrastructure/panel.py`) wraps it.

## Architecture & data flow

There is no framework — `flipdot.py:main()` is a single hand-written loop. Each iteration:
1. `Camera.read_frame()` → `image.crop()` to square.
2. Poll the Bluetooth controllers (`ControllerHub`) and feed presses through `ControllerInputBridge` into the `InputHub`.
3. Run MediaPipe pose (`human_pose.get_human_pose`) — **unless** it can be skipped: during sleep hours, while a controller drives a UI-only mode (`CONTROLLER_DRIVEN_UI_MODES`), or while a script runs. Pose is the loop's biggest cost, so skipping it keeps controller/script frames fast. Feed results into the `InputHub`.
4. `dispatch_actions(...)` applies queued input events (filtered by `mode_manager.get_allowed_input_sources()`) to the relevant mode.
5. `TransitionPolicy.apply(...)` decides the active mode from pose state and the clock (eyes visible + close enough → POSE; no pose for `POSE_TIMEOUT` → CLOCK; inside the sleep window → SLEEP) and returns a `TransitionState` (distance, angle, face-mesh results).
6. The `ModeRegistry` renders the active mode from a `RenderContext` → a `Frame`; `Panel.update(dots)` serializes and writes to hardware.
7. If the web UI is enabled, `WebServer.publish_frame(...)` mirrors the frame to browsers; the loop then sleeps/spin-waits to honor the per-mode FPS cap.

**The frame is the universal data type:** a `np.zeros((HEIGHT, WIDTH), dtype=np.uint8)` array of 0/1 dot values. Every renderer takes a `RenderContext` and returns one. Use the `Frame = np.ndarray` alias from `app/modes/contracts.py`. `Panel.WIDTH`/`Panel.HEIGHT` are the source of truth for dimensions — never hardcode them.

**Control sources.** `ModeManager` tracks both the active mode and the active *control source* (`CONTROL_GESTURE` vs `CONTROL_CONTROLLER`). Connecting a controller switches the source to controller; with none connected it falls back to gesture. `get_allowed_input_sources(include_web=True)` is what gates which queued `InputHub` events actually drive the display (`web` is always allowed). Each mode has a per-mode FPS cap in `ModeManager.MAX_FPS`; `get_fps_limit()` returns 30 for the first 5s of a mode for responsive transitions.

**Modes** (`ModeManager.MODE_*`): `sleep`, `clock`, `pose`, `menu`, `paint`, `caricature`, `percussion`, `autodrum`, `beatmirror`, `tetris`, `pong`, `tank`, `worldcup`, `board`, `font_preview`, `script`. To add a mode: define a `MODE_*` constant + `MAX_FPS` entry in `ModeManager`, construct it in `factory.create_mode_instances`, and register a renderer in `registry.build_mode_registry`.

## Web UI, AI control & scripting

These are opt-in subsystems layered on the core loop; none run unless enabled by env vars.

- **Web UI** (`ENABLE_WEB_UI=true`): a FastAPI server (`app/infrastructure/web_server.py`) started lazily *after* the first panel update (so cold web-stack import never delays first pixels). It mirrors the live frame to browsers over `/ws` (WebSocket) and `/api/frame`, accepts browser pointer/click/action input, and exposes REST APIs for the **board** mode (drawing, text/image objects, saved boards), sleep schedule, font preview, scripts, chat, and controller metrics. Pages: `/` (console), `/chat`, `/scripts`, `/font-grid`, `/controller-metrics`. Binds loopback (`WEB_UI_HOST=127.0.0.1`) by default.
- **MCP server** (`app/infrastructure/mcp_server.py`, `ENABLE_MCP=true` by default): a `FastMCP` instance exposing the display to AI agents as tools (`get_display` returns the panel as ASCII art, `set_mode`, board drawing, `run_script`, etc.). The same object backs two consumers: the in-UI chat (in-process) and the external HTTP `/mcp` endpoint. **`/mcp` is only mounted when `MCP_AUTH_TOKEN` is set** and is bearer-gated + DNS-rebinding-protected; the in-UI chat works without it.
- **In-UI Claude chat** (`app/infrastructure/chat.py`): the backend runs the agentic tool-use loop itself as the MCP client and streams NDJSON back to `/chat`. Needs `ANTHROPIC_API_KEY`. Default model `claude-opus-4-8` (`ANTHROPIC_MODEL` / per-request override; allowed: `claude-opus-4-8`, `claude-sonnet-5`, `claude-fable-5`). Token usage and cost are accumulated per turn and per session. Conversations persist to `state/chat_sessions/`.
- **Scripted animations** (`script` mode + `app/services/sandbox.py`): LLM- or user-authored Python frame generators (`setup`/`step` returning `(state, frame)`). Sandboxed in **four layers** — AST allow-list (only `numpy`/`math`/`random`), `bubblewrap` OS isolation (no network/filesystem, fails closed if `bwrap` is missing), a restricted-builtins subprocess, and rlimits + per-frame timeouts. Only a shape-checked `uint8` buffer crosses back. Tunable via `SANDBOX_*` env vars. Saved scripts live in `state/scripts/`.

## Project-specific conventions

- **Drawing = direct numpy indexing.** Set/XOR pixels by slicing the frame: `frame[row*8:row*8+7, 0:slice] ^= 1`. Menu rows are 8px tall (`row*8`). See `menu.py` `Checkbox.draw` and `clock.py` `update_frame`. For geometry (lines, circles, points) use the clipping-safe helpers in `app/services/draw.py` instead of hand-rolling Bresenham.
- **Text** uses a hand-coded bitmap font in `text.py`. Render with `text.write(frame, str, x=, y=, size=5|6, style=)`. Only glyphs present in the font tables exist — add new bitmaps there if needed.
- **Transitions** in `transition.py` operate on frames: `blend(a, b, alpha)`, `resolve`/`disolve(dots, alpha)` use random pixel masks (no greyscale — display is 1-bit).
- **Interaction is dwell-based**, not clicks: hover the right index finger over a `MenuItem` for `CLICK_TIME` (2s) to trigger `on_click`. `human_pose.is_arms_crossed` held for 2s opens the menu via `mode_manager.click_menu()`. With a controller connected, the same modes are driven by `ControllerInputBridge` instead (`toggle_menu`, buttons, etc.).
- Mode classes that need to switch modes take `mode_manager` in their constructor and call `mode_manager.set_mode(...)` (see `Menu`, `factory.create_mode_instances`).

## Code quality standards

These apply to all new and modified code; tooling enforces them (see "Quality gate" below).

**Documentation**
- Every module, public class, and public function/method has a docstring: a concise one-line imperative summary (e.g. "Decide the active mode from pose state."). Add `Args:`/`Returns:`/`Raises:` blocks only when the signature isn't self-explanatory. Private helpers (`_name`) may skip docstrings when the name and types are obvious.
- JS: document every exported/top-level function with a JSDoc block (`@param`, `@returns`). Trivial DOM-wiring one-liners may be skipped.
- Comments explain **why**, not what. Delete commented-out code rather than shipping it.

**Type hints (Python)**
- Annotate every function signature — all parameters and the return type (including `-> None`). Annotate non-obvious module/class attributes.
- Reuse the `Frame = np.ndarray` alias from `app/modes/contracts.py` instead of bare `np.ndarray`; add new aliases for other repeated shapes.
- Prefer precise types. Reserve `Any` for genuinely dynamic third-party objects (MediaPipe pose/face results). Use `Optional[...]` / `| None` for nullable values — pose results can be `None` (see Gotchas).

**Formatting & style**
- Python is formatted and linted by **Ruff** (line length 100, double quotes; rules pyflakes/pycodestyle/isort/pyupgrade/bugbear). Imports group stdlib / third-party / local. `flipPyDot` and `web_ui/vendor` are excluded.
- JS/CSS/HTML are formatted by **Prettier**; JS is linted by **ESLint**. `web_ui/vendor/` is excluded.
- No magic numbers in logic — name constants as class/module-level uppercase (follow `ModeManager.MAX_FPS`, `Paint.CLICK_TIME`). Never hardcode panel dimensions; use `Panel.WIDTH`/`Panel.HEIGHT`.

**Logging & errors**
- Use the module logger (`logger = logging.getLogger(__name__)`); never `print` in `app/`. Choose levels deliberately: DEBUG per-frame, INFO lifecycle, WARNING/ERROR for failures.
- Catch specific exceptions, not bare `except:`. Don't swallow errors silently — log with context.

**Structure & tests**
- Keep functions small and single-purpose; extract when a function does several things or exceeds ~50 lines.
- New behavior ships with a `tests/` test mirroring the source layout (`tests/core`, `tests/infrastructure`, `tests/modes`, `tests/services`). Tests must run without hardware, network, or model files (see `tests/conftest.py`). Browser-facing UI behavior is covered by Playwright specs in `web_ui/tests/`.

**Quality gate** — run before considering work done:
```bash
ruff check . && ruff format --check . && mypy app && pipenv run pytest
npx --prefix web_ui prettier --check web_ui && npx --prefix web_ui eslint web_ui
```
mypy is gradual (`disallow_untyped_defs=false`) and checks the `app` package. `pytest` enforces `--cov-fail-under=40` on `app`.

## Hardware & environment

- **Device:** NVIDIA Jetson Orin Nano (aarch64), JetPack R36.4.7, running Ubuntu with Python 3.10.
- **Hardware deps:** real serial flip-dot panel at `/dev/ttyUSB0` (57600 baud), a V4L2 webcam at `/dev/video0`, and (optionally) Bluetooth/HID game controllers read via `evdev`.
- Run with `PREVIEW=true` to use `flippydot`'s on-screen pygame preview instead of serial — essential for dev without hardware.
- **Config via `.env`** (loaded by `python-dotenv`). Common keys:
  - Core: `CAMERA_INDEX`, `PREVIEW`, `DEBUG`, `LOG_LEVEL`, `SLEEP_HOUR_START`/`SLEEP_HOUR_END`, `FOCAL_SCALE`.
  - Web/AI: `ENABLE_WEB_UI`, `WEB_UI_HOST`, `WEB_UI_PORT`, `WEB_UI_ALLOWED_ORIGINS`, `ENABLE_MCP`, `MCP_AUTH_TOKEN`, `MCP_ALLOWED_HOSTS`, `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`.
  - Controllers: `PRIMARY_CONTROLLER_ADDRESS`/`PRIMARY_CONTROLLER_NAME`, `SECONDARY_CONTROLLER_ADDRESS`. BLE link tuning applied after each connect: `CONTROLLER_SUPERVISION_TIMEOUT_MS` (default 2000; 0 disables the LE Connection Update), `CONTROLLER_CONN_MIN_INTERVAL_MS`/`CONTROLLER_CONN_MAX_INTERVAL_MS` (default 15/30) — a longer supervision timeout lets weak controllers ride through brief signal fades instead of dropping.
  - Integrations: `OPENWEATHER_API_KEY` (weather), `API_FOOTBALL_API_KEY` (worldcup), `PIXELLAB_API_KEY`/`OPENAI_API_KEY` (image generation).
  - Sandbox/models: `SANDBOX_MEM_MB`, `SANDBOX_CPU_SECONDS`, `SANDBOX_NPROC`, `SANDBOX_FRAME_TIMEOUT`, `SANDBOX_STARTUP_TIMEOUT`, `SANDBOX_MAX_SOURCE_BYTES`, `MEDIAPIPE_MODELS_DIR`, `POSE_MODEL`.
  - `DEBUG=true` overlays distance/angle text on the bottom rows; `LOG_LEVEL` controls logging verbosity (default `INFO`, set `DEBUG` for per-second performance logs).

## Installed software on the Jetson

- **Python 3.10** (`/usr/bin/python3`) — packages installed system-wide via `pip3`, **not pipenv**.
- **mediapipe 0.10.18** — uses the Tasks API (`mediapipe.tasks`) with CPU/XNNPACK delegate. Model files live in `~/flipdot/models/` on the Jetson (`MEDIAPIPE_MODELS_DIR` overrides; excluded from rsync and `.gitignore`). If model files are absent (e.g. on a dev machine) the code automatically falls back to the legacy `mp.solutions.pose` API, so local development works without them. GPU delegate is not available in the pip build; TensorRT 10.3.0 is installed but not yet wired up.
- **TensorRT 10.3.0** — pre-installed with JetPack; future path for GPU-accelerated pose inference.
- **bubblewrap (`bwrap`)** — required for the script sandbox; if absent, scripts refuse to run (fail closed).
- **opencv-python, pyserial, requests, pillow, python-dotenv, fastapi, uvicorn, python-multipart, evdev, mcp, anthropic** — installed via pip3.

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

- **Run locally (dev machine):** `PREVIEW=true python3 flipdot.py` (add `ENABLE_WEB_UI=true` to bring up the browser console on http://127.0.0.1:8000).
- **Run tests locally:** `pipenv run pytest` (Python, from repo root) and `npx --prefix web_ui playwright test` (browser UI). Tests run without hardware/network and avoid hard dependency on MediaPipe model files.
- **Lint / format / type-check:** `ruff check .` and `ruff format --check .` (Python), `mypy app` (types), `npx --prefix web_ui prettier --check web_ui` and `npx --prefix web_ui eslint web_ui` (JS). See "Code quality standards" for the full gate. `ruff check --fix .` and `ruff format .` auto-apply most fixes.
- **Run on device:** `sudo systemctl start flipdot.service`; the service auto-restarts on crash (`Restart=always`).
- **Deploy:** `./deploy.sh [--debug]` — rsyncs to `flipdot@flipdot:/home/flipdot/flipdot` (with `--delete`, but `.env`, `state/`, and `models/` are excluded), sets `DEBUG` in `.env`, ensures `state/` and `/var/log/flipdot` exist, pip-installs any missing web/AI deps (`python-multipart`, `mcp`, `anthropic`), installs the systemd units (`flipdot.service`, `flipdot-bluetooth-ertm.service`), the udev rule that disables the onboard Bluetooth radio (`ops/udev/99-flipdot-disable-onboard-bt.rules`, so the external UB500 Plus dongle is the sole adapter), and the logrotate config, then reloads and restarts the services. The Bluetooth unit disables ERTM, which otherwise causes multi-second input freezes with BR/EDR HID game controllers.
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
- Pose results can be `None` — guard `pose_results.pose_landmarks` before use, as the main loop does. Pose is also `None` whenever inference was skipped (sleep hours, controller-driven UI mode, running script).
- The web UI and `/mcp` bind to loopback by default; the external `/mcp` endpoint stays disabled until `MCP_AUTH_TOKEN` is set, even when `ENABLE_MCP` is true.
- Sandboxed scripts run a *separate* `python -c` worker (not `multiprocessing`) so they import only numpy — never the host's heavy stack. Changing that would drag ~1 GB of libraries into the worker and break its memory rlimit.
