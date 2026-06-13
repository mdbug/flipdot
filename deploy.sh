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
  --exclude='.env' \
  --exclude='.tetris_highscore' \
  --exclude='models/' \
  ./ "${REMOTE_HOST}:${REMOTE_DIR}"

ssh "${REMOTE_HOST}" "ENV_FILE=${REMOTE_DIR}/.env; touch \"\$ENV_FILE\"; if grep -q '^DEBUG=' \"\$ENV_FILE\"; then sed -i 's/^DEBUG=.*/DEBUG=${DEBUG_VALUE}/' \"\$ENV_FILE\"; else echo 'DEBUG=${DEBUG_VALUE}' >> \"\$ENV_FILE\"; fi; if ! grep -q '^LOG_LEVEL=' \"\$ENV_FILE\"; then echo 'LOG_LEVEL=INFO' >> \"\$ENV_FILE\"; fi"

ssh "${REMOTE_HOST}" "sudo mkdir -p /var/log/flipdot && sudo touch /var/log/flipdot/output.log /var/log/flipdot/error.log && sudo chown -R flipdot:flipdot /var/log/flipdot && sudo chmod 755 /var/log/flipdot"
ssh "${REMOTE_HOST}" "sudo install -m 644 ${REMOTE_DIR}/ops/systemd/flipdot.service /etc/systemd/system/flipdot.service"
ssh "${REMOTE_HOST}" "sudo install -m 644 ${REMOTE_DIR}/ops/logrotate/flipdot /etc/logrotate.d/flipdot"
ssh "${REMOTE_HOST}" "sudo systemctl daemon-reload && sudo systemctl restart flipdot.service && sudo systemctl --no-pager --full status flipdot.service | sed -n '1,20p'"