#!/bin/bash
rsync -avz --delete --exclude='.git' --exclude='__pycache__' --exclude='.env' ./ flipdot@flipdot:/home/flipdot/flipdot
# if parameter --debug is passed, set env variable DEBUG to true
if [[ "$1" == "--debug" ]]; then
  ssh flipdot 'echo "DEBUG=true" >> /home/flipdot/flipdot/.env'
else
  ssh flipdot 'echo "DEBUG=false" >> /home/flipdot/flipdot/.env'
fi
echo flipdot | ssh -tt flipdot 'sudo systemctl restart flipdot.service'