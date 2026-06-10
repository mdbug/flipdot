#!/bin/bash
rsync -avz --delete --exclude='.git' --exclude='__pycache__' --exclude='.env' --exclude='models/' ./ flipdot@flipdot:/home/flipdot/flipdot
# if parameter --debug is passed, set env variable DEBUG to true
DEBUG_VALUE=false
if [[ "$1" == "--debug" ]]; then
  DEBUG_VALUE=true
fi

ssh flipdot "ENV_FILE=/home/flipdot/flipdot/.env; if grep -q '^DEBUG=' \"\$ENV_FILE\"; then sed -i 's/^DEBUG=.*/DEBUG=${DEBUG_VALUE}/' \"\$ENV_FILE\"; else echo 'DEBUG=${DEBUG_VALUE}' >> \"\$ENV_FILE\"; fi"
ssh flipdot 'sudo systemctl restart flipdot.service'