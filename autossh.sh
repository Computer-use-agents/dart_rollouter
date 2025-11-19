#!/bin/bash
# ssh root@112.30.139.26 -p 52921
LUCHEN_USER="root"
LUCHEN_HOST="112.30.139.26"
LUCHEN_PORT="52921"   # 提到的端口

# autossh -M 0 -fNg \
#   -p ${LUCHEN_PORT} \
#   -R 0.0.0.0:14959:localhost:15959 \
#   -R 0.0.0.0:3206:localhost:3306 \
#   ${LUCHEN_USER}@${LUCHEN_HOST}



# REMOTE_USER="lipengxiang"
# REMOTE_HOST="115.190.219.88"
# PASSWORD="lipengxiang"

while true
do
    echo "Starting SSH tunnel..."

    # sshpass -p "$PASSWORD" ssh \
    sshpass  ssh \
        -p ${LUCHEN_PORT} \
      -o ServerAliveInterval=10 \
      -o ServerAliveCountMax=3 \
      -o StrictHostKeyChecking=no \
      -fNg \
      -R 0.0.0.0:3206:localhost:3306 \
      ${LUCHEN_USER}@${LUCHEN_HOST}

    echo "SSH tunnel closed. Reconnecting in 600 seconds..."
    sleep 600
done
