#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-1}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/resource_logs}"
LOG_FILE="${1:-$LOG_DIR/resource_$(date '+%Y%m%d_%H%M%S').csv}"
PROCESS_MATCH="${PROCESS_MATCH:-ml_self_play}"

mkdir -p "$(dirname "$LOG_FILE")"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "error: nvidia-smi not found; cannot log GPU stats" >&2
  exit 1
fi

echo "Logging resource usage every ${INTERVAL_SECONDS}s"
echo "Output: $LOG_FILE"
echo "Process match: $PROCESS_MATCH"
echo "Press Ctrl-C to stop."

echo "timestamp,cpu_user_pct,cpu_system_pct,cpu_idle_pct,cpu_iowait_pct,mem_used_mib,mem_available_mib,gpu_util_pct,gpu_mem_used_mib,gpu_mem_free_mib,gpu_temp_c,gpu_power_w,gpu_graphics_clock_mhz,trainer_pid,trainer_cpu_pct,trainer_mem_pct,trainer_rss_kib" > "$LOG_FILE"

while true; do
  timestamp="$(date '+%F %T')"

  cpu_line="$(top -bn1 | awk '/^%Cpu/ && $2 ~ /^[0-9.]+/ {printf "%s,%s,%s,%s", $2, $4, $8, $10; exit}')"
  if [[ -z "$cpu_line" ]]; then
    cpu_line=",,,";
  fi

  mem_line="$(free -m | awk '/^Mem:/ {printf "%s,%s", $3, $7; exit}')"
  if [[ -z "$mem_line" ]]; then
    mem_line=",";
  fi

  gpu_line="$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free,temperature.gpu,power.draw,clocks.gr --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
  if [[ -z "$gpu_line" ]]; then
    gpu_line=",,,,,";
  fi

  trainer_line="$(ps -C "$PROCESS_MATCH" -o pid=,%cpu=,%mem=,rss= --sort=-%cpu 2>/dev/null | awk 'NR==1 {gsub(/^ +| +$/, ""); gsub(/ +/, ","); print; exit}' || true)"
  if [[ -z "$trainer_line" ]]; then
    trainer_line=",,,";
  fi

  row="$timestamp,$cpu_line,$mem_line,$gpu_line,$trainer_line"
  echo "$row" | tee -a "$LOG_FILE"

  sleep "$INTERVAL_SECONDS"
done
