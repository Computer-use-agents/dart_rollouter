#!/usr/bin/env bash
# start.sh - Start run_model first, then run run.py after it's ready; exit run_model after run.py finishes
# Usage: chmod +x start.sh; execute as container entrypoint
source miniconda3/bin/activate 
conda activate verl
cd /workspace/codes/verl/rollouter/
set -euo pipefail

mkdir -p /root/verl/rollouter/gpu_util_log
MONITOR_ID="/root/verl/rollouter/gpu_util_log/gpu_monitor_$(date +%Y%m%d_%H%M%S)_$$"

# GPU monitoring diagnostics - Check if nvidia-smi is available, don't exit on failure (GPU may be unavailable)
echo "[INFO] Attempting to start GPU monitoring..."
if command -v nvidia-smi >/dev/null 2>&1; then
  # Try to run nvidia-smi, if NVML error occurs, log but don't exit
  if timeout 5 nvidia-smi > /dev/null 2>&1; then
    nohup nvidia-smi --query-gpu=index,timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1 > "${MONITOR_ID}.csv" 2>&1 &
    echo "[INFO] GPU monitoring started"
  else
    echo "[WARN] nvidia-smi execution failed, skipping GPU monitoring (might be container environment or driver issue)"
  fi
else
  echo "[WARN] nvidia-smi not found, skipping GPU monitoring"
fi

# ===== Configurable parameters (support environment variable override) =====
PORT="${PORT:-15959}"
HEALTH_ENDPOINT="${HEALTH_ENDPOINT:-http://127.0.0.1:${PORT}}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-600}"     # Wait for run_model to be ready timeout (seconds)
POLL_INTERVAL="${POLL_INTERVAL:-10}"           # Health check polling interval (seconds)
TERMINATE_GRACE_SECONDS="${TERMINATE_GRACE_SECONDS:-5}"  # Graceful wait for stopping model service (seconds)

export PYTHONUNBUFFERED=1

# Tool check
if ! command -v curl >/dev/null 2>&1; then
  echo "[FATAL] curl is required for health checks" >&2
  exit 127
fi

MODEL_PID=""
RUN_PID=""

# ===== Signal forwarding: When container terminates, forward signal to run.py =====
forward_to_run() {
  echo "[INFO] Termination signal received, forwarding to run.py ..."
  if [[ -n "${RUN_PID}" ]] && kill -0 "${RUN_PID}" 2>/dev/null; then
    kill -TERM "${RUN_PID}" 2>/dev/null || true
  fi
}
trap forward_to_run INT TERM

# ===== EXIT cleanup: Stop model service when finishing =====
cleanup_model() {
  if [[ -n "${MODEL_PID}" ]] && kill -0 "${MODEL_PID}" 2>/dev/null; then
    echo "[INFO] Stopping model service (PID=${MODEL_PID})"
    kill "${MODEL_PID}" 2>/dev/null || true
    # Wait for graceful exit
    for ((i=0; i<TERMINATE_GRACE_SECONDS*10; i++)); do
      kill -0 "${MODEL_PID}" 2>/dev/null || break
      sleep 0.1
    done
    # Force kill if still running
    if kill -0 "${MODEL_PID}" 2>/dev/null; then
      echo "[WARN] Model service did not exit within ${TERMINATE_GRACE_SECONDS}s, sending SIGKILL"
      kill -9 "${MODEL_PID}" 2>/dev/null || true
    fi
  fi
}
trap cleanup_model EXIT

# ===== 1) Start model service (background) =====
echo "[INFO] Starting model service: python src/run_model.py"
python src/run_model.py &
MODEL_PID=$!
echo "[INFO] Model service started, PID=${MODEL_PID}"

# ===== 2) Poll health check until ready or timeout =====
echo "[INFO] Waiting for model service to be ready at ${HEALTH_ENDPOINT} (timeout ${STARTUP_TIMEOUT}s)..."
SECONDS=0
while true; do
  # Fail immediately if service process has exited
  if ! kill -0 "${MODEL_PID}" 2>/dev/null; then
    echo "[ERROR] Model service process exited prematurely!"
    exit 1
  fi
  # Health check
  if RESP="$(curl -fsS "${HEALTH_ENDPOINT}" 2>/dev/null || true)"; then
    if grep -q '"Model Service is running."' <<<"${RESP}"; then
      echo "[INFO] Health check passed: ${RESP}"
      break
    fi
  fi
  (( SECONDS >= STARTUP_TIMEOUT )) && { echo "[ERROR] Model service not ready within timeout"; exit 1; }
  sleep "${POLL_INTERVAL}"
done

# ===== 3) Run main program (foreground wait), clean up model service after finish =====
echo "[INFO] Starting main program: python src/run.py $*"
python -u src/run.py "$@" &
RUN_PID=$!

# Wait for run.py to finish, record exit code
wait "${RUN_PID}"; RUN_EXIT=$?
echo "[INFO] run.py exited with status code=${RUN_EXIT}"

# Script ending will trigger EXIT trap, thus stopping model service
exit "${RUN_EXIT}"