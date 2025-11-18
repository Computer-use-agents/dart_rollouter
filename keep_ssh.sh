#!/bin/bash

REMOTE_USER="lipengxiang"
REMOTE_HOST="115.190.219.88"
PASSWORD="lipengxiang"

while true
do
    echo "Starting SSH tunnel..."

    # sshpass -p "$PASSWORD" ssh \
    sshpass  ssh \
      -o ServerAliveInterval=10 \
      -o ServerAliveCountMax=3 \
      -o StrictHostKeyChecking=no \
      -fNg \
      -L 0.0.0.0:15959:localhost:15959 \
      -L 0.0.0.0:3306:localhost:3306 \
      ${REMOTE_USER}@${REMOTE_HOST}

    echo "SSH tunnel closed. Reconnecting in 600 seconds..."
    sleep 600
done
