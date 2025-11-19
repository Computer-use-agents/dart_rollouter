#!/bin/bash

LUCHEN_USER="root"
LUCHEN_HOST="112.30.139.26"
LUCHEN_PORT="52082"   # 提到的端口

autossh -M 0 -fNg \
  -p ${LUCHEN_PORT} \
  -L 0.0.0.0:14959:localhost:15959 \
  -L 0.0.0.0:3206:localhost:3306 \
  ${LUCHEN_USER}@${LUCHEN_HOST}
