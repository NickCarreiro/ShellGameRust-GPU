#!/usr/bin/env bash
# Attempt to recover NVIDIA compute after a CUDA launch/Xid failure without rebooting.
#
# By default this prompts for sudo when it is attached to a terminal. Set
# GPU_RECOVERY_ALLOW_PROMPT=0 for unattended jobs that should fail fast instead.

set -u

GPU_ID="${GPU_ID:-0}"
GPU_RECOVERY_ALLOW_PROMPT="${GPU_RECOVERY_ALLOW_PROMPT:-1}"
SUDO_CMD=()
KDE_STATS_STOPPED=0

prepare_sudo() {
  SUDO_CMD=()
  if ! command -v sudo >/dev/null 2>&1; then
    echo "GPU recovery: sudo is not installed or not on PATH." >&2
    return 1
  fi

  if sudo -n true 2>/dev/null; then
    SUDO_CMD=(sudo -n)
    return 0
  fi

  if [[ "$GPU_RECOVERY_ALLOW_PROMPT" == "1" && -t 0 ]]; then
    echo
    echo "GPU recovery: sudo credentials are needed for driver reset/reload."
    echo "GPU recovery: prompting once; this keeps CUDA recovery no-reboot but not silent."
    if sudo -v; then
      SUDO_CMD=(sudo)
      return 0
    fi
    echo "GPU recovery: sudo authentication failed or was cancelled." >&2
    return 1
  fi

  echo "GPU recovery: sudo requires a password, but interactive prompting is disabled/unavailable." >&2
  return 1
}

run_with_sudo() {
  "${SUDO_CMD[@]}" "$@"
}

module_loaded() {
  lsmod | awk '{print $1}' | grep -qx "$1"
}

stop_system_monitor_stats() {
  KDE_STATS_STOPPED=0

  if command -v systemctl >/dev/null 2>&1 \
      && systemctl --user is-active --quiet plasma-ksystemstats.service 2>/dev/null; then
    echo
    echo "GPU recovery: stopping KDE system monitor stats backend"
    systemctl --user stop plasma-ksystemstats.service 2>/dev/null || true
    KDE_STATS_STOPPED=1
    sleep 0.5
    return 0
  fi

  if pgrep -u "$(id -u)" -x ksystemstats >/dev/null 2>&1; then
    echo
    echo "GPU recovery: stopping ksystemstats process"
    pkill -u "$(id -u)" -x ksystemstats 2>/dev/null || true
    KDE_STATS_STOPPED=1
    sleep 0.5
  fi
}

restart_system_monitor_stats() {
  if command -v systemctl >/dev/null 2>&1; then
    if [[ "$KDE_STATS_STOPPED" == "1" ]] \
        || systemctl --user list-unit-files plasma-ksystemstats.service >/dev/null 2>&1; then
      echo
      echo "GPU recovery: restarting KDE system monitor stats backend"
      systemctl --user restart plasma-ksystemstats.service 2>/dev/null || true
      return 0
    fi
  fi

  if [[ "$KDE_STATS_STOPPED" == "1" ]] && command -v ksystemstats >/dev/null 2>&1; then
    echo
    echo "GPU recovery: starting ksystemstats"
    nohup ksystemstats >/tmp/ksystemstats-restart.log 2>&1 &
  fi
}

stop_stale_user_nvidia_monitors() {
  local pids=""
  pids="$(pgrep -u "$(id -u)" -f 'nvidia-smi dmon' 2>/dev/null || true)"
  if [[ -z "$pids" ]]; then
    return 0
  fi

  echo
  echo "GPU recovery: stopping stale user NVIDIA monitor(s):"
  echo "$pids" | sed 's/^/  pid /'
  # nvidia-smi dmon is only a sampler, but it can keep /dev/nvidia-uvm open and
  # block the UVM reload that clears a poisoned CUDA compute context.
  kill $pids 2>/dev/null || true
  sleep 0.5
  for pid in $pids; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
  done
}

kill_gpu_users() {
  local pids
  pids="$(fuser /dev/nvidia* 2>/dev/null | tr ' ' '\n' | sort -u | grep -v "^[[:space:]]*$" || true)"
  # Exclude our own shell PID so we don't suicide.
  pids="$(echo "$pids" | grep -v "^$$\$" || true)"
  if [[ -z "$pids" ]]; then
    echo "GPU recovery: no GPU users to kill."
    return 0
  fi
  echo "GPU recovery: sending SIGTERM to GPU users: $(echo "$pids" | tr '\n' ' ')"
  for pid in $pids; do
    run_with_sudo kill -15 "$pid" 2>/dev/null || kill -15 "$pid" 2>/dev/null || true
  done
  sleep 4
  local survivors=""
  for pid in $pids; do
    if kill -0 "$pid" 2>/dev/null; then
      survivors="$survivors $pid"
    fi
  done
  if [[ -n "$survivors" ]]; then
    echo "GPU recovery: sending SIGKILL to survivors:$survivors"
    for pid in $survivors; do
      run_with_sudo kill -9 "$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null || true
    done
    sleep 1
  fi
}

echo "GPU recovery: checking NVIDIA state for GPU $GPU_ID..."
nvidia-smi || {
  echo "GPU recovery: nvidia-smi is not responding; a full driver reload or reboot may be required." >&2
  exit 1
}

# Obtain sudo credentials upfront so they're cached before any step needs them.
# This avoids a mid-recovery password prompt that can be missed when output is scrolling.
prepare_sudo

echo
echo "GPU recovery: driver-reported recovery action:"
RECOVERY_ACTION="$(
  nvidia-smi -i "$GPU_ID" --query-gpu=gpu_recovery_action --format=csv,noheader,nounits 2>/dev/null \
    | head -n 1 \
    | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
)"
if [[ -n "$RECOVERY_ACTION" ]]; then
  if [[ "$RECOVERY_ACTION" =~ ^Field[[:space:]].*not.*valid ]]; then
    echo "  unavailable on this driver/GPU"
  else
    echo "  $RECOVERY_ACTION"
  fi
  if [[ "$RECOVERY_ACTION" =~ [Rr]eboot ]]; then
    echo "GPU recovery: NVIDIA reports that a reboot is required; refusing to pretend a reset will be foolproof." >&2
    exit 1
  fi
else
  echo "  unavailable on this driver/GPU"
fi

echo
echo "GPU recovery: active compute clients:"
nvidia-smi -i "$GPU_ID" --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null \
  || echo "  no queryable compute clients"

echo
echo "GPU recovery: recent NVIDIA kernel faults:"
(dmesg --ctime 2>/dev/null || journalctl -k --since "2 hours ago" --no-pager 2>/dev/null) \
  | grep -Ei "NVRM|Xid|CUDA|nvidia" \
  | tail -n 12 || true

echo
echo "GPU recovery: active NVIDIA device users:"
fuser -v /dev/nvidia* 2>&1 || true
stop_stale_user_nvidia_monitors

echo
echo "GPU recovery: trying nvidia-smi GPU reset..."
if nvidia-smi --gpu-reset -i "$GPU_ID"; then
  echo "GPU recovery: nvidia-smi reset completed."
  restart_system_monitor_stats
  nvidia-smi || true
  exit 0
fi

if [[ ${#SUDO_CMD[@]} -gt 0 ]]; then
  SUDO_LABEL="${SUDO_CMD[*]}"
  stop_system_monitor_stats

  echo
  echo "GPU recovery: trying $SUDO_LABEL modprobe nvidia_uvm reload..."
  UVM_UNLOADED=0
  if run_with_sudo modprobe -r nvidia_uvm; then
    UVM_UNLOADED=1
  elif module_loaded nvidia_uvm; then
    echo "GPU recovery: nvidia_uvm is still loaded; reload is blocked, probably by an active compute client." >&2
  else
    echo "GPU recovery: nvidia_uvm was not loaded; attempting a fresh load." >&2
    UVM_UNLOADED=1
  fi

  if [[ "$UVM_UNLOADED" == "1" ]] && run_with_sudo modprobe nvidia_uvm; then
    echo "GPU recovery: nvidia_uvm reload completed."
    restart_system_monitor_stats
    nvidia-smi || true
    exit 0
  fi

  echo
  echo "GPU recovery: trying $SUDO_LABEL nvidia-smi GPU reset..."
  if run_with_sudo nvidia-smi --gpu-reset -i "$GPU_ID"; then
    echo "GPU recovery: nvidia-smi reset completed."
    restart_system_monitor_stats
    nvidia-smi || true
    exit 0
  fi
fi

restart_system_monitor_stats

echo
echo "GPU recovery: gentle recovery failed. Escalating: killing GPU users."
echo

stop_system_monitor_stats
kill_gpu_users
sleep 1

echo
echo "GPU recovery: retrying nvidia_uvm reload..."
UVM_RETRY=0
if run_with_sudo modprobe -r nvidia_uvm 2>/dev/null; then
  UVM_RETRY=1
elif ! module_loaded nvidia_uvm; then
  UVM_RETRY=1
fi
if [[ "$UVM_RETRY" == "1" ]] && run_with_sudo modprobe nvidia_uvm; then
  echo "GPU recovery: nvidia_uvm reload completed after killing GPU users."
  restart_system_monitor_stats
  nvidia-smi || true
  exit 0
fi

echo
echo "GPU recovery: retrying nvidia-smi GPU reset..."
if run_with_sudo nvidia-smi --gpu-reset -i "$GPU_ID"; then
  echo "GPU recovery: nvidia-smi reset completed after killing GPU users."
  restart_system_monitor_stats
  nvidia-smi || true
  exit 0
fi

echo
echo "GPU recovery: recovery not possible without reboot. Rebooting in 5 seconds..."
echo "GPU recovery: press Ctrl+C to cancel."
sleep 5
run_with_sudo reboot
