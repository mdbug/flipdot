#!/bin/bash
set -euo pipefail

REMOTE_HOST="flipdot"
REMOTE_DIR="/home/flipdot/flipdot"
DEBUG_VALUE=false

# If parameter --debug is passed, set env variable DEBUG to true.
if [[ "${1:-}" == "--debug" ]]; then
  DEBUG_VALUE=true
fi

rsync -avz --delete \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='.pytest_cache' \
  --exclude='.env' \
  --exclude='state/' \
  --exclude='models/' \
  ./ "${REMOTE_HOST}:${REMOTE_DIR}"

ssh "${REMOTE_HOST}" "REMOTE_DIR='${REMOTE_DIR}' DEBUG_VALUE='${DEBUG_VALUE}' bash -s" <<'EOF'
set -euo pipefail

ENV_FILE="${REMOTE_DIR}/.env"
touch "${ENV_FILE}"
if grep -q '^DEBUG=' "${ENV_FILE}"; then
  sed -i "s/^DEBUG=.*/DEBUG=${DEBUG_VALUE}/" "${ENV_FILE}"
else
  echo "DEBUG=${DEBUG_VALUE}" >> "${ENV_FILE}"
fi
if ! grep -q '^LOG_LEVEL=' "${ENV_FILE}"; then
  echo 'LOG_LEVEL=INFO' >> "${ENV_FILE}"
fi

mkdir -p "${REMOTE_DIR}/state"

sudo mkdir -p /var/log/flipdot
sudo touch /var/log/flipdot/output.log /var/log/flipdot/error.log
sudo chown -R flipdot:flipdot /var/log/flipdot
sudo chmod 755 /var/log/flipdot

if ! python3 - <<'PY'
import importlib.util
import sys

sys.exit(0 if importlib.util.find_spec('multipart') else 1)
PY
then
  python3 -m pip install --user --disable-pip-version-check python-multipart
fi

if ! python3 - <<'PY'
import importlib.util
import sys

sys.exit(0 if importlib.util.find_spec('mcp') else 1)
PY
then
  python3 -m pip install --user --disable-pip-version-check mcp
fi

if ! python3 - <<'PY'
import importlib.util
import sys

sys.exit(0 if importlib.util.find_spec('anthropic') else 1)
PY
then
  python3 -m pip install --user --disable-pip-version-check anthropic
fi

daemon_reload_needed=false
if ! sudo cmp -s "${REMOTE_DIR}/ops/systemd/flipdot.service" /etc/systemd/system/flipdot.service; then
  sudo install -m 644 "${REMOTE_DIR}/ops/systemd/flipdot.service" /etc/systemd/system/flipdot.service
  daemon_reload_needed=true
fi

if ! sudo cmp -s "${REMOTE_DIR}/ops/systemd/flipdot-bluetooth-ertm.service" /etc/systemd/system/flipdot-bluetooth-ertm.service; then
  sudo install -m 644 "${REMOTE_DIR}/ops/systemd/flipdot-bluetooth-ertm.service" /etc/systemd/system/flipdot-bluetooth-ertm.service
  daemon_reload_needed=true
fi

if ! sudo cmp -s "${REMOTE_DIR}/ops/logrotate/flipdot" /etc/logrotate.d/flipdot; then
  sudo install -m 644 "${REMOTE_DIR}/ops/logrotate/flipdot" /etc/logrotate.d/flipdot
fi

# Disable the onboard Bluetooth radio so the external UB500 Plus dongle is the
# sole adapter (better antenna; avoids two-radio contention).
if ! sudo cmp -s "${REMOTE_DIR}/ops/udev/99-flipdot-disable-onboard-bt.rules" /etc/udev/rules.d/99-flipdot-disable-onboard-bt.rules; then
  sudo install -m 644 "${REMOTE_DIR}/ops/udev/99-flipdot-disable-onboard-bt.rules" /etc/udev/rules.d/99-flipdot-disable-onboard-bt.rules
  sudo udevadm control --reload
fi

if [[ "${daemon_reload_needed}" == "true" ]]; then
  sudo systemctl daemon-reload
fi

# Disable Bluetooth ERTM to prevent multi-second input freezes with
# Xbox-compatible HID controllers such as the IINE mini controllers.
sudo systemctl enable --now flipdot-bluetooth-ertm.service
sudo systemctl restart flipdot-bluetooth-ertm.service
sudo systemctl restart flipdot.service
sudo systemctl --no-pager --full status flipdot.service | sed -n '1,20p'
EOF
