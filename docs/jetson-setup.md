# Jetson setup

How to set up an NVIDIA Jetson Orin Nano from scratch to run the flipdot installation. The repo assumes this exact target — JetPack R36 (Ubuntu 22.04, Python 3.10, aarch64) — but everything degrades gracefully on other machines (CPU pose inference, pygame preview instead of the panel).

## 1. Hardware

- **NVIDIA Jetson Orin Nano** with JetPack R36.x.
- **Flip-dot panel:** 4× AlfaZeta XY5 28×7 modules, chained on RS-485, connected through a USB serial adapter → shows up as `/dev/ttyUSB0` (57600 baud).
- **Webcam:** any V4L2 UVC camera → `/dev/video0`.
- **Bluetooth dongle (for controllers):** [TP-Link UB500 Plus](https://www.tp-link.com/de/home-networking/adapter/ub500-plus/) (Realtek RTL8761B), ideally on a USB extension cable placed *outside* the enclosure near the play area. The Jetson's onboard Bluetooth radio is too weak for the low-power controllers (buried in the enclosure it reads ~15 dB worse) and is disabled during deploy — see [Bluetooth](#5-bluetooth--controllers).
- **Controllers (optional):** 2× [IINE GameBrick Mini](https://iine.store/products/iine-gamebrick-mini-retro-controller) (BLE, D-pad + A/B).

## 2. OS user & access

Create a `flipdot` user with sudo rights and give it access to serial, video, and input devices:

```bash
sudo adduser flipdot
sudo usermod -aG sudo,dialout,video,input flipdot
```

On your dev machine, add an SSH alias so `deploy.sh` (which targets `flipdot@flipdot`) works:

```
# ~/.ssh/config
Host flipdot
    HostName <jetson-ip-or-hostname>
    User flipdot
```

## 3. Python dependencies

Packages are installed **system-wide with pip3** on the device (pipenv is only used on dev machines):

```bash
sudo apt update
sudo apt install -y python3-pip bubblewrap v4l-utils
pip3 install opencv-python pyserial requests pillow python-dotenv \
             fastapi uvicorn python-multipart evdev mcp anthropic openai
```

Notes:

- **`bubblewrap` is required** for the script sandbox (AI/user-authored animations). Scripts refuse to run without it — the sandbox fails closed.
- `deploy.sh` re-checks the web/AI deps (`python-multipart`, `mcp`, `anthropic`, `openai`) on every deploy and installs any that are missing.

### MediaPipe (pose detection)

Two options:

- **GPU wheel (recommended, ~5× faster pose inference — 42 ms vs 208 ms/frame):** a custom GPU-enabled build for exactly this platform (JetPack R36.4, Python 3.10) is published at [mdbug/mediapipe-jetson-gpu](https://github.com/mdbug/mediapipe-jetson-gpu). Install the wheel from the latest release. It uses a headless EGL context, so it works under systemd with no display attached; first startup takes ~9 s longer for one-time shader compilation.
- **Stock CPU wheel:** `pip3 install mediapipe==0.10.18`. Same code path — `human_pose.py` requests the GPU delegate first and falls back to CPU automatically.

### Model files

The MediaPipe task models are not in the repo and are excluded from deploy — download them once:

```bash
mkdir -p ~/flipdot/models && cd ~/flipdot/models
wget -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
wget -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
wget -q https://storage.googleapis.com/mediapipe-models/image_segmenter/hair_segmenter/float32/latest/hair_segmenter.tflite
```

If the models are absent, the code falls back to the legacy `mp.solutions.pose` API (and skips hair rendering in caricature mode) — useful on dev machines, not what you want in production.

## 4. Deploy

From the repo root on your dev machine:

```bash
./deploy.sh            # add --debug to enable the on-panel debug overlay
```

This rsyncs the tree to `flipdot@flipdot:/home/flipdot/flipdot` (excluding `.env`, `state/`, `models/`), ensures `state/` and `/var/log/flipdot` exist, installs missing web/AI pip deps, installs the systemd units, the udev rule, and the logrotate config, then restarts the services:

- **`flipdot.service`** — the main loop. Runs as user `flipdot`, `Restart=always`, logs to `/var/log/flipdot/{output,error}.log`. It carries `AmbientCapabilities=CAP_NET_ADMIN` so controller link metrics (RSSI etc.) can be read via `btmgmt`.
- **`flipdot-bluetooth-ertm.service`** — disables Bluetooth ERTM before `bluetooth.service` starts. ERTM causes multi-second input stalls with BR/EDR HID controllers (e.g. Xbox pads); the BLE IINE pads don't need it, but it's kept as insurance.
- **`99-flipdot-disable-onboard-bt.rules`** (udev) — deauthorizes the onboard Realtek radio (USB `13d3:3549`) at boot so the UB500 Plus is the sole adapter (`hci0`). Delete the rule and reboot to restore the onboard radio.

## 5. Configuration (`.env`)

`.env` lives on the device at `/home/flipdot/flipdot/.env` and is **never touched by rsync** — edit it over SSH. Minimal production example:

```bash
CAMERA_INDEX=0
SLEEP_HOUR_START=0
SLEEP_HOUR_END=7

ENABLE_WEB_UI=true
WEB_UI_HOST=0.0.0.0        # loopback by default; open it up only on a trusted LAN

# Bluetooth controllers (see pairing below; unset = controllers disabled)
PRIMARY_CONTROLLER_ADDRESS=AA:BB:CC:DD:EE:01
PRIMARY_CONTROLLER_NAME=IINE_keyboard
SECONDARY_CONTROLLER_ADDRESS=AA:BB:CC:DD:EE:02

# Integrations (all optional)
ANTHROPIC_API_KEY=...       # in-UI AI chat
OPENWEATHER_API_KEY=...     # weather on the clock face
WEATHER_CITY=Berlin
WEATHER_COUNTRY_CODE=DE
```

The annotated full list of variables is in [.env.example](../.env.example) at the repo root.

## 6. Bluetooth & controllers

Pairing the IINE GameBrick Minis (they are BLE "Just-Works" devices, and a bit particular):

```bash
sudo systemctl stop flipdot.service    # its reconnect loop interferes with pairing

bluetoothctl
  agent NoInputNoOutput                # the default agent fails with AuthenticationFailed
  default-agent
  scan on                              # press a button on the controller to wake it
  # it advertises under a *random* address, discoverable by name: IINE_keyboard
  pair <address>
  trust <address>
  connect <address>
```

Two quirks worth knowing:

- **The controllers regenerate their BLE address on every re-pair.** After pairing, put the *bonded* address into `.env` (`PRIMARY_CONTROLLER_ADDRESS=...`). You can find the bonded devices under `/var/lib/bluetooth/<adapter-mac>/`.
- They sleep aggressively; a button press wakes them. The `CONTROLLER_SUPERVISION_TIMEOUT_MS` / `CONTROLLER_CONN_*` env vars tune the BLE link so weak signal fades don't drop the connection.

Verify signal quality on the **Metrics** page of the web UI (`/controller-metrics`) — with the UB500 Plus positioned well, expect roughly −75…−85 dBm from the pads.

## 7. Operate

```bash
sudo systemctl status flipdot.service      # is it running?
sudo systemctl restart flipdot.service     # restart after config changes
sudo tail -f /var/log/flipdot/output.log   # app logs
sudo tail -f /var/log/flipdot/error.log    # tracebacks
```

After boot you should see the clock on the panel within seconds (pose model loading adds a few more before person detection kicks in). If `ENABLE_WEB_UI=true`, the console is at `http://<jetson>:8000/`.

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| Panel dark, service running | Wrong serial device — check `ls /dev/ttyUSB*` and that the user is in `dialout` |
| `Camera initialized` missing in logs | Wrong `CAMERA_INDEX` — check `v4l2-ctl --list-devices` |
| Scripts fail with "worker died during startup" | `bubblewrap` missing, or sandbox rlimits too tight (`SANDBOX_*` vars) |
| Controllers pair but drop constantly | Onboard radio still active (check `hciconfig`/udev rule), dongle inside the enclosure, or ERTM re-enabled |
| Pose detection slow (~200 ms/frame) | Stock CPU MediaPipe wheel installed — see the GPU wheel above |
| Blank controller metrics | Service missing `CAP_NET_ADMIN` (use the shipped unit file) |
