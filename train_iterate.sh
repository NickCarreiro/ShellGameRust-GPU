#!/usr/bin/env bash
# Iterative refinement from a saved checkpoint.
#
# Usage:
#   ./train_iterate.sh [checkpoint_dir] [iterations]
#
# Defaults to the strongest available core checkpoint under models/core/.
# Tie-break rule for equal EV tags: newest directory mtime wins.
#
# Each iteration:
#   1. Resume coagent training from previous self_play_models.json
#   2. Promote best_evader + final searcher into self_play_models.json
#   3. Evaluate the promoted bundle (the exact one used for next iteration)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_BIN="$ROOT_DIR/target/release/ml_self_play"
TRAIN_CHILD_RUNNING=0
RESOURCE_LOGGER_PID=""
USER_STOP_REQUESTED=0

stop_resource_logger() {
  local pid="${RESOURCE_LOGGER_PID:-}"
  if [[ -n "$pid" ]]; then
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
    RESOURCE_LOGGER_PID=""
  fi
}

handle_interrupt() {
  if [[ "${TRAIN_CHILD_RUNNING:-0}" == "1" ]]; then
    # Signal passes to the training binary, which sets its graceful-stop flag and
    # finishes the current generation before exiting. Print once so the user knows
    # to wait — pressing Ctrl+C again is harmless (the flag is already set).
    if [[ "$USER_STOP_REQUESTED" == "0" ]]; then
      USER_STOP_REQUESTED=1
      echo
      echo "Stop signal sent — trainer will save and exit after the current generation. No further iterations will run."
    fi
    return 0
  fi
  stop_resource_logger
  echo
  echo "Interrupted"
  exit 130
}
trap handle_interrupt INT

dir_has_entries() {
  [[ -d "$1" ]] && [[ -n "$(find "$1" -mindepth 1 -maxdepth 1 -print -quit)" ]]
}

# --- AUTO-SELECT CHECKPOINT LOGIC ---
DEFAULT_CORE_DIR="$ROOT_DIR/models/core"
if [[ -z "${1:-}" ]]; then
    echo "Checkpoint selection"
    echo "  mode:       auto"
    echo "  search dir: $DEFAULT_CORE_DIR"

    best_dir=""
    best_ev=-1
    best_mtime=-1

    if [[ -d "$DEFAULT_CORE_DIR" ]]; then
      for candidate in "$DEFAULT_CORE_DIR"/*; do
        [[ -d "$candidate" ]] || continue
        [[ -f "$candidate/self_play_models.json" ]] || continue

        dir_name="$(basename "$candidate")"
        # Prefer reading peak evader score from training_history.json so this works
        # with any directory name, not just ones with an embedded ev tag.
        ev_score=""
        if [[ -f "$candidate/training_history.json" ]]; then
          ev_score="$(python3 - "$candidate/training_history.json" 2>/dev/null <<'PYEOF'
import json, sys
try:
    h = json.load(open(sys.argv[1]))
    best = max(int(g.get("evader_score", -9999)) for g in h)
    print(best)
except Exception:
    pass
PYEOF
)"
        fi
        # Fall back to the ev tag embedded in the directory name for backward compatibility.
        if [[ -z "$ev_score" ]] && [[ "$dir_name" =~ ([0-9]+)ev ]]; then
          ev_score="${BASH_REMATCH[1]}"
        fi
        [[ -n "$ev_score" ]] || continue

        mtime="$(stat -c %Y "$candidate" 2>/dev/null || echo 0)"
        if (( ev_score > best_ev || (ev_score == best_ev && mtime > best_mtime) )); then
          best_ev=$ev_score
          best_mtime=$mtime
          best_dir="$candidate"
        fi
      done
    fi

    if [[ -n "$best_dir" ]]; then
        AUTO_CHECKPOINT="$best_dir"
        echo "  selected:   $(basename "$best_dir")"
        echo "  ev tag:     $best_ev"
        echo "  tie break:  newest among ties"
    else
        # Fallback: latest numeric checkpoint/vN that has a bundle.
        fallback_root="$ROOT_DIR/models/checkpoints"
        fallback_dir=""
        fallback_ver=-1
        if [[ -d "$fallback_root" ]]; then
          for candidate in "$fallback_root"/v*; do
            [[ -d "$candidate" ]] || continue
            [[ -f "$candidate/self_play_models.json" ]] || continue
            base="$(basename "$candidate")"
            if [[ "$base" =~ ^v([0-9]+)$ ]]; then
              ver="${BASH_REMATCH[1]}"
              if (( ver > fallback_ver )); then
                fallback_ver=$ver
                fallback_dir="$candidate"
              fi
            fi
          done
        fi

        if [[ -n "$fallback_dir" ]]; then
          AUTO_CHECKPOINT="$fallback_dir"
          echo "  selected:   $(basename "$fallback_dir")"
          echo "  reason:     no EV-tagged core checkpoints found"
        else
          echo "ERROR: Could not auto-select a checkpoint with self_play_models.json" >&2
          exit 1
        fi
    fi
fi

CHECKPOINT_DIR="${1:-$AUTO_CHECKPOINT}"
ITERATIONS="${2:-3}"
# ------------------------------------

if ! [[ "$ITERATIONS" =~ ^[0-9]+$ ]]; then
  echo "ERROR: iterations must be a positive integer, got: '$ITERATIONS'" >&2
  if [[ "$ITERATIONS" == *=* ]]; then
    echo "Hint: env vars must be prefixed before the command." >&2
    echo "  Correct: RUST_BACKTRACE=1 ./train_iterate.sh \"$CHECKPOINT_DIR\" 1" >&2
  fi
  exit 1
fi
if (( ITERATIONS < 1 )); then
  echo "ERROR: iterations must be >= 1, got: '$ITERATIONS'" >&2
  exit 1
fi

if [[ ! -f "$CHECKPOINT_DIR/self_play_models.json" ]]; then
  echo "ERROR: missing model bundle: $CHECKPOINT_DIR/self_play_models.json" >&2
  exit 1
fi

GENERATIONS="${GENERATIONS:-12}"
EPISODES_PER_EVAL="${EPISODES_PER_EVAL:-36}"

_POPULATION_EXPLICIT="${POPULATION_SIZE+x}"
_MIN_NODES_EXPLICIT="${MIN_NODES+x}"
_MAX_NODES_EXPLICIT="${MAX_NODES+x}"

# Inherit curriculum from checkpoint's training history so switching to a new
# run doesn't silently reset population/node range back to script defaults.
# Explicit env vars (POPULATION_SIZE=N, MIN_NODES=N, MAX_NODES=N) take priority.
_CKPT_POP="" _CKPT_MIN="" _CKPT_MAX=""
if [[ -f "$CHECKPOINT_DIR/training_history.json" ]]; then
  _CKPT_CURRICULUM="$(python3 - "$CHECKPOINT_DIR/training_history.json" 2>/dev/null <<'PYEOF'
import json, sys
try:
    h = json.load(open(sys.argv[1]))
    last = h[0]
    p, mn, mx = last.get("population_size"), last.get("min_nodes"), last.get("max_nodes")
    if p and mn and mx:
        print(p, mn, mx)
except Exception:
    pass
PYEOF
)"
  if [[ "$_CKPT_CURRICULUM" =~ ^[0-9]+[[:space:]]+[0-9]+[[:space:]]+[0-9]+$ ]]; then
    read -r _CKPT_POP _CKPT_MIN _CKPT_MAX <<<"$_CKPT_CURRICULUM"
    echo "Curriculum inherited from checkpoint"
    echo "  population: ${_CKPT_POP}"
    echo "  nodes:      ${_CKPT_MIN}..${_CKPT_MAX}"
    echo "  (override with POPULATION_SIZE=N MIN_NODES=N MAX_NODES=N)"
  fi
fi

POPULATION_SIZE="${POPULATION_SIZE:-${_CKPT_POP:-50}}"
MIN_NODES="${MIN_NODES:-${_CKPT_MIN:-30}}"
MAX_NODES="${MAX_NODES:-${_CKPT_MAX:-50}}"
CURRICULUM_POPULATION_CAP="${CURRICULUM_POPULATION_CAP:-75}"
CURRICULUM_MAX_NODES_CAP="${CURRICULUM_MAX_NODES_CAP:-65}"
if [[ -z "$_POPULATION_EXPLICIT" && -n "$CURRICULUM_POPULATION_CAP" && "$POPULATION_SIZE" =~ ^[0-9]+$ && "$CURRICULUM_POPULATION_CAP" =~ ^[0-9]+$ && "$POPULATION_SIZE" -gt "$CURRICULUM_POPULATION_CAP" ]]; then
  echo "Curriculum velocity cap"
  echo "  population: $POPULATION_SIZE -> $CURRICULUM_POPULATION_CAP"
  echo "  (override with POPULATION_SIZE=N or CURRICULUM_POPULATION_CAP=N)"
  POPULATION_SIZE="$CURRICULUM_POPULATION_CAP"
fi
if [[ -z "$_MAX_NODES_EXPLICIT" && -n "$CURRICULUM_MAX_NODES_CAP" && "$MAX_NODES" =~ ^[0-9]+$ && "$CURRICULUM_MAX_NODES_CAP" =~ ^[0-9]+$ && "$MAX_NODES" -gt "$CURRICULUM_MAX_NODES_CAP" ]]; then
  old_min="$MIN_NODES"
  old_max="$MAX_NODES"
  span=$((MAX_NODES - MIN_NODES))
  MAX_NODES="$CURRICULUM_MAX_NODES_CAP"
  if [[ -z "$_MIN_NODES_EXPLICIT" ]]; then
    MIN_NODES=$((MAX_NODES - span))
    if (( MIN_NODES < 1 )); then MIN_NODES=1; fi
  fi
  if (( MIN_NODES > MAX_NODES )); then MIN_NODES="$MAX_NODES"; fi
  echo "Curriculum velocity cap"
  echo "  nodes:      ${old_min}..${old_max} -> ${MIN_NODES}..${MAX_NODES}"
  echo "  (override with MIN_NODES=N MAX_NODES=N or CURRICULUM_MAX_NODES_CAP=N)"
fi
MAX_ATTEMPTS_FACTOR="${MAX_ATTEMPTS_FACTOR:-4}"
MAX_ATTEMPTS_RATIO="${MAX_ATTEMPTS_RATIO:-0.40}"
# Leave cap unset by default; ratio-based budget scales naturally with node count.
MAX_ATTEMPTS_CAP="${MAX_ATTEMPTS_CAP:-}"
GPU_SCORE_BATCH_CELLS="${GPU_SCORE_BATCH_CELLS:-4096}"
GPU_SCORE_BATCH_ROWS="${GPU_SCORE_BATCH_ROWS:-}"
GPU_SINGLE_SCORE_ROWS="${GPU_SINGLE_SCORE_ROWS:-2048}"
HALL_OF_FAME_SIZE="${HALL_OF_FAME_SIZE:-8}"
HALL_SAMPLE_COUNT="${HALL_SAMPLE_COUNT:-1}"
STATIC_OPPONENT_SAMPLE_COUNT="${STATIC_OPPONENT_SAMPLE_COUNT:-2}"
MUTATION_SCALE="${MUTATION_SCALE:-0.18}"
ES_LR="${ES_LR:-0.008}"
SEARCHER_LR_SCALE="${SEARCHER_LR_SCALE:-0.25}"
SEARCHER_UPDATE_INTERVAL="${SEARCHER_UPDATE_INTERVAL:-2}"
SEARCHER_MAX_FOUND_RATE="${SEARCHER_MAX_FOUND_RATE:-0.55}"
SEARCHER_MAX_FOUND_RATE_JUMP="${SEARCHER_MAX_FOUND_RATE_JUMP:-0.25}"
# Default patience tracks generation count so early-stopping is meaningful.
PATIENCE="${PATIENCE:-$GENERATIONS}"
# Adaptive curriculum: when evader score stagnates, widen the tree range and
# spend a larger ES population before patience gives up on the run.
STAGNATION_GROW_AFTER="${STAGNATION_GROW_AFTER:-0}"
STAGNATION_NODE_STEP="${STAGNATION_NODE_STEP:-5}"
STAGNATION_POPULATION_STEP="${STAGNATION_POPULATION_STEP:-25}"
STAGNATION_MAX_NODES_CAP="${STAGNATION_MAX_NODES_CAP:-}"
STAGNATION_POPULATION_CAP="${STAGNATION_POPULATION_CAP:-}"
BASE_SEED="${SEED:-9999}"
EVAL_EPISODES="${EVAL_EPISODES:-100}"
USE_CUDA="${USE_CUDA:-1}"
RECOVER_ON_TRAIN_FAILURE="${RECOVER_ON_TRAIN_FAILURE:-1}"
AUTO_GPU_RECOVER="${AUTO_GPU_RECOVER:-1}"
# Let recover_gpu.sh prompt for sudo when train_iterate.sh is run from a TTY.
GPU_RECOVERY_ALLOW_PROMPT="${GPU_RECOVERY_ALLOW_PROMPT:-1}"
GPU_RECOVERY_SCRIPT="${GPU_RECOVERY_SCRIPT:-$ROOT_DIR/recover_gpu.sh}"
CUDA_FAILURE_CPU_FALLBACK="${CUDA_FAILURE_CPU_FALLBACK:-0}"
MAX_TRAIN_RESTARTS="${MAX_TRAIN_RESTARTS:-3}"
MAX_EVAL_RESTARTS="${MAX_EVAL_RESTARTS:-2}"
CUDA_HEALTH_CHECK="${CUDA_HEALTH_CHECK:-1}"
CUDA_HEALTH_BIN="$ROOT_DIR/target/release/cuda_health"
RESOURCE_LOGGING="${RESOURCE_LOGGING:-1}"
RESOURCE_LOG_INTERVAL="${RESOURCE_LOG_INTERVAL:-0.2}"
RESOURCE_LOG_DIR="${RESOURCE_LOG_DIR:-$ROOT_DIR/resource_logs}"
RESOURCE_LOG_SCRIPT="${RESOURCE_LOG_SCRIPT:-$ROOT_DIR/log_resources.sh}"
RESOURCE_PROCESS_MATCH="${RESOURCE_PROCESS_MATCH:-ml_self_play}"
CUDA_DRIVER_POISONED=0
export GPU_RECOVERY_ALLOW_PROMPT

CARGO_FEATURE_ARGS=()
if [[ "$USE_CUDA" == "0" ]]; then
  CARGO_FEATURE_ARGS=("--no-default-features")
  export SHELLGAME_FORCE_CPU=1
else
  if [[ -z "${NVCC_CCBIN:-}" ]] && command -v gcc-13 &>/dev/null; then
    export NVCC_CCBIN=/usr/bin/gcc-13
  fi
fi

build_train_binary() {
  cargo build --release "$@" --bin ml_self_play
}

build_cuda_health_binary() {
  if [[ "$USE_CUDA" != "0" && "$CUDA_HEALTH_CHECK" == "1" ]]; then
    cargo build --release --bin cuda_health
  fi
}

cuda_failure_in_log() {
  local log_path="$1"
  grep -Eqi 'CUDA_ERROR|DriverError\(CUDA|unspecified launch failure|cudarc|cuda_backend|CudaSlice|GPU .*failed|CUDA .*failed' "$log_path"
}

# Full CUDA recovery: gentle modprobe/reset → kill GPU users → reboot.
# Sets CUDA_DRIVER_POISONED=0 on success, 1 on failure.
recover_cuda_failure() {
  local gpu_id="${GPU_ID:-0}"
  local -a _sudo=()

  # Acquire sudo upfront so it's ready before any step needs it.
  if sudo -n true 2>/dev/null; then
    _sudo=(sudo -n)
  elif [[ "${GPU_RECOVERY_ALLOW_PROMPT:-1}" == "1" && -t 0 ]]; then
    echo "GPU recovery: sudo credentials needed for driver reset."
    if sudo -v 2>/dev/null; then
      _sudo=(sudo)
    fi
  fi

  # Kill nvidia-smi dmon samplers — they hold /dev/nvidia-uvm open and block
  # the UVM module unload even when no compute job is running.
  local _dmon_pids
  _dmon_pids="$(pgrep -u "$(id -u)" -f 'nvidia-smi dmon' 2>/dev/null || true)"
  if [[ -n "$_dmon_pids" ]]; then
    echo "GPU recovery: stopping nvidia-smi dmon monitor(s): $(echo "$_dmon_pids" | tr '\n' ' ')"
    kill $_dmon_pids 2>/dev/null || true
    sleep 0.5
    for _dp in $_dmon_pids; do kill -0 "$_dp" 2>/dev/null && kill -9 "$_dp" 2>/dev/null || true; done
  fi

  # Stop KDE system monitor — it can also hold nvidia_uvm open and block unload.
  local _kde_stopped=0
  if command -v systemctl >/dev/null 2>&1 \
      && systemctl --user is-active --quiet plasma-ksystemstats.service 2>/dev/null; then
    systemctl --user stop plasma-ksystemstats.service 2>/dev/null || true
    _kde_stopped=1
    sleep 0.3
  fi

  _gpu_restart_kde() {
    if [[ "$_kde_stopped" == "1" ]] && command -v systemctl >/dev/null 2>&1; then
      systemctl --user restart plasma-ksystemstats.service 2>/dev/null || true
    fi
  }

  # ── Gentle: nvidia_uvm reload ──────────────────────────────────────────────
  echo "GPU recovery: trying nvidia_uvm reload..."
  local _uvm_clear=0
  if [[ ${#_sudo[@]} -gt 0 ]]; then
    if "${_sudo[@]}" modprobe -r nvidia_uvm 2>/dev/null; then
      _uvm_clear=1
    elif ! lsmod | awk '{print $1}' | grep -qx nvidia_uvm; then
      _uvm_clear=1
    fi
    if (( _uvm_clear )) && "${_sudo[@]}" modprobe nvidia_uvm 2>/dev/null; then
      echo "GPU recovery: nvidia_uvm reload ok"
      _gpu_restart_kde; CUDA_DRIVER_POISONED=0; return 0
    fi
  fi

  # ── Gentle: nvidia-smi GPU reset ──────────────────────────────────────────
  echo "GPU recovery: trying nvidia-smi GPU reset..."
  if nvidia-smi --gpu-reset -i "$gpu_id" 2>/dev/null \
      || { [[ ${#_sudo[@]} -gt 0 ]] && "${_sudo[@]}" nvidia-smi --gpu-reset -i "$gpu_id" 2>/dev/null; }; then
    echo "GPU recovery: nvidia-smi reset ok"
    _gpu_restart_kde; CUDA_DRIVER_POISONED=0; return 0
  fi

  # ── Escalate: kill all GPU users ───────────────────────────────────────────
  echo "GPU recovery: gentle recovery failed — killing GPU users"
  local _pids
  _pids="$(fuser /dev/nvidia* 2>/dev/null | tr ' ' '\n' | sort -u \
    | grep -vE "^[[:space:]]*$|^$$\$" || true)"
  if [[ -n "$_pids" ]]; then
    echo "  SIGTERM: $(echo "$_pids" | tr '\n' ' ')"
    for _pid in $_pids; do
      { [[ ${#_sudo[@]} -gt 0 ]] && "${_sudo[@]}" kill -15 "$_pid" 2>/dev/null; } \
        || kill -15 "$_pid" 2>/dev/null || true
    done
    sleep 4
    for _pid in $_pids; do
      if kill -0 "$_pid" 2>/dev/null; then
        echo "  SIGKILL: $_pid"
        { [[ ${#_sudo[@]} -gt 0 ]] && "${_sudo[@]}" kill -9 "$_pid" 2>/dev/null; } \
          || kill -9 "$_pid" 2>/dev/null || true
      fi
    done
    sleep 1
  fi

  # ── Retry after kill ───────────────────────────────────────────────────────
  echo "GPU recovery: retrying after killing GPU users..."
  _uvm_clear=0
  if [[ ${#_sudo[@]} -gt 0 ]]; then
    if "${_sudo[@]}" modprobe -r nvidia_uvm 2>/dev/null; then
      _uvm_clear=1
    elif ! lsmod | awk '{print $1}' | grep -qx nvidia_uvm; then
      _uvm_clear=1
    fi
    if (( _uvm_clear )) && "${_sudo[@]}" modprobe nvidia_uvm 2>/dev/null; then
      echo "GPU recovery: nvidia_uvm reload ok after kill"
      _gpu_restart_kde; CUDA_DRIVER_POISONED=0; return 0
    fi
    if "${_sudo[@]}" nvidia-smi --gpu-reset -i "$gpu_id" 2>/dev/null; then
      echo "GPU recovery: nvidia-smi reset ok after kill"
      _gpu_restart_kde; CUDA_DRIVER_POISONED=0; return 0
    fi
  fi

  # ── Last resort: reboot ───────────────────────────────────────────────────
  _gpu_restart_kde
  echo "GPU recovery: reboot required — rebooting in 5 seconds (Ctrl+C to cancel)"
  sleep 5
  if [[ ${#_sudo[@]} -gt 0 ]]; then
    "${_sudo[@]}" reboot
  else
    reboot
  fi
  CUDA_DRIVER_POISONED=1
  return 1
}

recover_gpu_if_enabled() {
  if [[ "$AUTO_GPU_RECOVER" != "1" || "$USE_CUDA" == "0" ]]; then
    echo "WARNING: automatic GPU recovery is disabled or CUDA is not active; not retrying unsafe CUDA." >&2
    CUDA_DRIVER_POISONED=1
    return 1
  fi
  echo "GPU recovery"
  echo "  action: gentle → kill GPU users → reboot"
  if recover_cuda_failure; then
    echo "  status: completed"
    CUDA_DRIVER_POISONED=0
    return 0
  else
    echo "WARNING: GPU recovery did not complete." >&2
    CUDA_DRIVER_POISONED=1
    return 1
  fi
}

cuda_health_check_once() {
  local log_path="$1"
  local health_population="${CUDA_HEALTH_POPULATION:-${TRAIN_POPULATION_SIZE:-$POPULATION_SIZE}}"
  local health_rows="${CUDA_HEALTH_ROWS:-}"
  if [[ -z "$health_rows" ]]; then
    health_rows=$((GPU_SCORE_BATCH_CELLS / health_population))
    if (( health_rows < 1 )); then
      health_rows=1
    fi
  fi
  CUDA_LAUNCH_BLOCKING=1 \
    REQUIRE_CUDA=1 \
    CUDA_HEALTH_POPULATION="$health_population" \
    CUDA_HEALTH_ROWS="$health_rows" \
    CUDA_HEALTH_SINGLE_ROWS="${CUDA_HEALTH_SINGLE_ROWS:-$GPU_SINGLE_SCORE_ROWS}" \
    "$CUDA_HEALTH_BIN" 2>&1 | tee "$log_path"
  return "${PIPESTATUS[0]}"
}

run_train_binary() {
  if [[ "$USE_CUDA" == "0" ]]; then
    "$TRAIN_BIN" "$@"
  else
    CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}" REQUIRE_CUDA=1 "$TRAIN_BIN" "$@"
  fi
}

start_resource_logger() {
  local log_path="$1"

  stop_resource_logger
  if [[ "$RESOURCE_LOGGING" != "1" ]]; then
    return 0
  fi
  if [[ ! -x "$RESOURCE_LOG_SCRIPT" ]]; then
    echo "WARNING: resource logger script is missing or not executable: $RESOURCE_LOG_SCRIPT" >&2
    return 0
  fi

  mkdir -p "$(dirname "$log_path")"
  echo "    resource log: $log_path"
  INTERVAL_SECONDS="$RESOURCE_LOG_INTERVAL" \
    LOG_DIR="$(dirname "$log_path")" \
    PROCESS_MATCH="$RESOURCE_PROCESS_MATCH" \
    "$RESOURCE_LOG_SCRIPT" "$log_path" > "${log_path%.csv}.logger.log" 2>&1 &
  RESOURCE_LOGGER_PID=$!
}

ensure_cuda_healthy() {
  local label="$1"
  if [[ "$USE_CUDA" == "0" || "$CUDA_HEALTH_CHECK" != "1" ]]; then
    return 0
  fi
  if [[ "$CUDA_DRIVER_POISONED" == "1" ]]; then
    echo "CUDA health"
    echo "  label:   $label"
    echo "  status:  skipped; driver context is still marked poisoned after recovery failure"
    return 1
  fi

  local log_path="$2"
  local attempts="$3"
  for ((health_try = 1; health_try <= attempts; health_try++)); do
    echo "CUDA health"
    echo "  label:   $label"
    echo "  attempt: $health_try/$attempts"
    echo "  log:     $log_path"
    set +e
    cuda_health_check_once "$log_path"
    local health_status=$?
    set -e
    if (( health_status == 0 )); then
      CUDA_DRIVER_POISONED=0
      return 0
    fi
    if ! cuda_failure_in_log "$log_path"; then
      return "$health_status"
    fi
    if (( health_try >= attempts )); then
      CUDA_DRIVER_POISONED=1
      return "$health_status"
    fi
    if ! recover_gpu_if_enabled; then
      echo "CUDA health"
      echo "  status: recovery failed; refusing to relaunch CUDA into a poisoned driver context" >&2
      return "$health_status"
    fi
  done
}

switch_training_to_cpu() {
  if [[ "$USE_CUDA" == "0" ]]; then
    return 0
  fi
  echo "CUDA fallback"
  echo "  action: rebuilding trainer CPU-only and continuing without touching CUDA"
  USE_CUDA=0
  export SHELLGAME_FORCE_CPU=1
  CARGO_FEATURE_ARGS=("--no-default-features")
  build_train_binary "${CARGO_FEATURE_ARGS[@]}"
}

read_training_recovery_state() {
  local out_dir="$1"
  python3 - "$out_dir" <<'PYEOF' || true
import json
import sys

out_dir = sys.argv[1]
try:
    with open(f"{out_dir}/training_history.json") as fh:
        history = json.load(fh)
    if not history:
        raise RuntimeError("empty history")
    last = history[-1]
    generation = int(last.get("generation", 0))
    population = int(last.get("population_size", 0))
    min_nodes = int(last.get("min_nodes", 0))
    max_nodes = int(last.get("max_nodes", 0))
    print(f"{generation} {population} {min_nodes} {max_nodes}")
except Exception:
    pass
PYEOF
}

ATTEMPT_FLAG_ARGS=(
  --max-attempts-factor "$MAX_ATTEMPTS_FACTOR"
)
if [[ -n "$MAX_ATTEMPTS_RATIO" ]]; then
  ATTEMPT_FLAG_ARGS+=(--max-attempts-ratio "$MAX_ATTEMPTS_RATIO")
fi
if [[ -n "$MAX_ATTEMPTS_CAP" ]]; then
  ATTEMPT_FLAG_ARGS+=(--max-attempts-cap "$MAX_ATTEMPTS_CAP")
fi
export GPU_SCORE_BATCH_CELLS
export GPU_SINGLE_SCORE_ROWS
if [[ -n "$GPU_SCORE_BATCH_ROWS" ]]; then
  export GPU_SCORE_BATCH_ROWS
fi

ADAPTIVE_GROWTH_ARGS=()
if [[ -n "$STAGNATION_GROW_AFTER" && "$STAGNATION_GROW_AFTER" != "0" ]]; then
  ADAPTIVE_GROWTH_ARGS+=(
    --stagnation-grow-after "$STAGNATION_GROW_AFTER"
    --stagnation-node-step "$STAGNATION_NODE_STEP"
    --stagnation-population-step "$STAGNATION_POPULATION_STEP"
  )
  if [[ -n "$STAGNATION_MAX_NODES_CAP" ]]; then
    ADAPTIVE_GROWTH_ARGS+=(--stagnation-max-nodes-cap "$STAGNATION_MAX_NODES_CAP")
  fi
  if [[ -n "$STAGNATION_POPULATION_CAP" ]]; then
    ADAPTIVE_GROWTH_ARGS+=(--stagnation-population-cap "$STAGNATION_POPULATION_CAP")
  fi
fi

CURRENT_POPULATION_SIZE="$POPULATION_SIZE"
CURRENT_MIN_NODES="$MIN_NODES"
CURRENT_MAX_NODES="$MAX_NODES"

echo
echo "Iterative refinement"
echo "  checkpoint:        $CHECKPOINT_DIR"
echo "  iterations:        $ITERATIONS"
echo "  generations:       $GENERATIONS"
echo "  population:        $POPULATION_SIZE"
echo "  patience:          $PATIENCE"
echo "  episodes/eval:     $EPISODES_PER_EVAL"
echo "  nodes:             ${MIN_NODES}..${MAX_NODES}"
echo "  curriculum caps:   population=${CURRICULUM_POPULATION_CAP:-none} max_nodes=${CURRICULUM_MAX_NODES_CAP:-none}"
echo "  attempt budget:    factor=$MAX_ATTEMPTS_FACTOR ratio=${MAX_ATTEMPTS_RATIO:-none} cap=${MAX_ATTEMPTS_CAP:-none}"
echo "  es lr:             $ES_LR"
echo "  searcher throttle: lr_scale=$SEARCHER_LR_SCALE update_every=$SEARCHER_UPDATE_INTERVAL found_rate_cap=$SEARCHER_MAX_FOUND_RATE max_jump=$SEARCHER_MAX_FOUND_RATE_JUMP"
echo "  gpu score batch:   cells=$GPU_SCORE_BATCH_CELLS rows=${GPU_SCORE_BATCH_ROWS:-auto} single_rows=$GPU_SINGLE_SCORE_ROWS"
if [[ ${#ADAPTIVE_GROWTH_ARGS[@]} -gt 0 ]]; then
  echo "  adaptive growth:   after=$STAGNATION_GROW_AFTER node_step=$STAGNATION_NODE_STEP population_step=$STAGNATION_POPULATION_STEP"
  echo "                     node_cap=${STAGNATION_MAX_NODES_CAP:-none} population_cap=${STAGNATION_POPULATION_CAP:-none}"
else
  echo "  adaptive growth:   disabled"
fi
echo "  train recovery:    $RECOVER_ON_TRAIN_FAILURE"
echo "  gpu recovery:      auto=$AUTO_GPU_RECOVER"
echo "                     script=$GPU_RECOVERY_SCRIPT"
echo "                     sudo_prompt=$GPU_RECOVERY_ALLOW_PROMPT"
echo "  cuda restart:      train_restarts=$MAX_TRAIN_RESTARTS eval_restarts=$MAX_EVAL_RESTARTS"
echo "  cuda fallback:     cpu_on_failure=$CUDA_FAILURE_CPU_FALLBACK"
echo "  cuda health:       preflight=$CUDA_HEALTH_CHECK"
echo "  resource logging:  enabled=$RESOURCE_LOGGING interval=${RESOURCE_LOG_INTERVAL}s dir=$RESOURCE_LOG_DIR"
echo

echo "Build"
echo "  binary: $TRAIN_BIN"
build_train_binary "${CARGO_FEATURE_ARGS[@]}"
build_cuda_health_binary
echo

CURRENT_CHECKPOINT="$CHECKPOINT_DIR"

for ((iter = 1; iter <= ITERATIONS; iter++)); do
  ITER_SEED=$((BASE_SEED + (iter - 1) * 1117))

  # Derive next version number from checkpoint directory name
  CURR_NAME="$(basename "$CURRENT_CHECKPOINT")"
  CURR_VER="${CURR_NAME#v}"         # strip leading 'v'
  if [[ "$CURR_VER" =~ ^[0-9]+$ ]]; then
    NEXT_VER=$((CURR_VER + 1))
    NEXT_DIR="$ROOT_DIR/models/checkpoints/v${NEXT_VER}"
  else
    NEXT_DIR="${CURRENT_CHECKPOINT}_iter${iter}"
  fi

  if dir_has_entries "$NEXT_DIR"; then
    BASE_NEXT_DIR="$NEXT_DIR"
    retry=1
    while dir_has_entries "${BASE_NEXT_DIR}_retry${retry}"; do
      retry=$((retry + 1))
    done
    NEXT_DIR="${BASE_NEXT_DIR}_retry${retry}"
    echo "Output directory"
    echo "  requested: $(basename "$BASE_NEXT_DIR")"
    echo "  status:    already contains files"
    echo "  using:     $NEXT_DIR"
  fi

  mkdir -p "$NEXT_DIR"

  echo "Iteration $iter / $ITERATIONS"
  echo "  output:     $(basename "$NEXT_DIR")"
  echo "  seed:       $ITER_SEED"
  echo "  curriculum: population=$CURRENT_POPULATION_SIZE nodes=${CURRENT_MIN_NODES}..${CURRENT_MAX_NODES}"

  TRAIN_STATUS=1
  TRAIN_ATTEMPT=1
  TRAIN_GENERATIONS_REMAINING="$GENERATIONS"
  TRAIN_RESUME_BUNDLE="$CURRENT_CHECKPOINT/self_play_models.json"
  TRAIN_POPULATION_SIZE="$CURRENT_POPULATION_SIZE"
  TRAIN_MIN_NODES="$CURRENT_MIN_NODES"
  TRAIN_MAX_NODES="$CURRENT_MAX_NODES"

  while :; do
    TRAIN_LOG="$NEXT_DIR/train_attempt${TRAIN_ATTEMPT}.log"
    TRAIN_HEALTH_LOG="$NEXT_DIR/cuda_health_train_attempt${TRAIN_ATTEMPT}.log"
    TRAIN_RESOURCE_LOG="$RESOURCE_LOG_DIR/$(basename "$NEXT_DIR")_train_attempt${TRAIN_ATTEMPT}.csv"
    echo "  launch:"
    echo "    attempt:     $TRAIN_ATTEMPT/$MAX_TRAIN_RESTARTS"
    echo "    backend:     $([[ "$USE_CUDA" == "0" ]] && echo "cpu" || echo "cuda")"
    echo "    generations: $TRAIN_GENERATIONS_REMAINING"
    echo "    population:  $TRAIN_POPULATION_SIZE"
    echo "    nodes:       ${TRAIN_MIN_NODES}..${TRAIN_MAX_NODES}"
    echo "    resume:      $TRAIN_RESUME_BUNDLE"
    echo "    log:         $TRAIN_LOG"

    if ensure_cuda_healthy "train attempt $TRAIN_ATTEMPT" "$TRAIN_HEALTH_LOG" "$MAX_TRAIN_RESTARTS"; then
      :
    else
      TRAIN_STATUS=$?
      echo "CUDA health failed before training launch"
      echo "  log: $TRAIN_HEALTH_LOG"
      if [[ "$CUDA_FAILURE_CPU_FALLBACK" == "1" ]]; then
        switch_training_to_cpu
        continue
      fi
      break
    fi

    set +e
    TRAIN_CHILD_RUNNING=1
    start_resource_logger "$TRAIN_RESOURCE_LOG"
    run_train_binary train \
      --training-mode coagent \
      --generations "$TRAIN_GENERATIONS_REMAINING" \
      --population-size "$TRAIN_POPULATION_SIZE" \
      --episodes-per-eval "$EPISODES_PER_EVAL" \
      --min-nodes "$TRAIN_MIN_NODES" \
      --max-nodes "$TRAIN_MAX_NODES" \
      "${ATTEMPT_FLAG_ARGS[@]}" \
      --hall-of-fame-size "$HALL_OF_FAME_SIZE" \
      --hall-sample-count "$HALL_SAMPLE_COUNT" \
      --static-opponent-sample-count "$STATIC_OPPONENT_SAMPLE_COUNT" \
      --mutation-scale "$MUTATION_SCALE" \
      --es-lr "$ES_LR" \
      --searcher-lr-scale "$SEARCHER_LR_SCALE" \
      --searcher-update-interval "$SEARCHER_UPDATE_INTERVAL" \
      --searcher-max-found-rate "$SEARCHER_MAX_FOUND_RATE" \
      --searcher-max-found-rate-jump "$SEARCHER_MAX_FOUND_RATE_JUMP" \
      --patience "$PATIENCE" \
      "${ADAPTIVE_GROWTH_ARGS[@]}" \
      --seed "$ITER_SEED" \
      --resume-from "$TRAIN_RESUME_BUNDLE" \
      --output-dir "$NEXT_DIR" 2>&1 | (trap '' INT; tee "$TRAIN_LOG")
    TRAIN_STATUS=${PIPESTATUS[0]}
    TRAIN_CHILD_RUNNING=0
    stop_resource_logger
    set -e

    if (( TRAIN_STATUS == 0 )); then
      break
    fi

    if [[ "$RECOVER_ON_TRAIN_FAILURE" != "1" ]]; then
      break
    fi

    if ! cuda_failure_in_log "$TRAIN_LOG"; then
      break
    fi

    echo
    echo "CUDA failure detected"
    echo "  status: $TRAIN_STATUS"
    echo "  log:    $TRAIN_LOG"

    if [[ ! -f "$NEXT_DIR/self_play_models.json" ]]; then
      echo "  recovery: no checkpoint bundle exists yet; cannot resume this attempt" >&2
      break
    fi

    RECOVERY_STATE="$(read_training_recovery_state "$NEXT_DIR")"
    if [[ ! "$RECOVERY_STATE" =~ ^[0-9]+[[:space:]]+[0-9]+[[:space:]]+[0-9]+[[:space:]]+[0-9]+$ ]]; then
      echo "  recovery: could not read training_history.json state; cannot compute remaining generations" >&2
      break
    fi

    read -r RECOVERED_GEN RECOVERED_POPULATION RECOVERED_MIN_NODES RECOVERED_MAX_NODES <<<"$RECOVERY_STATE"
    echo "  recovery checkpoint:"
    echo "    generation: $RECOVERED_GEN"
    echo "    population: $RECOVERED_POPULATION"
    echo "    nodes:      ${RECOVERED_MIN_NODES}..${RECOVERED_MAX_NODES}"

    if (( RECOVERED_GEN >= TRAIN_GENERATIONS_REMAINING )); then
      echo "  recovery: checkpoint already reached requested generation count; continuing pipeline"
      TRAIN_STATUS=0
      break
    fi

    if (( TRAIN_ATTEMPT >= MAX_TRAIN_RESTARTS )); then
      echo "  recovery: max restarts reached; preserving checkpoint and continuing with best available model" >&2
      break
    fi

    if recover_gpu_if_enabled; then
      :
    elif [[ "$CUDA_FAILURE_CPU_FALLBACK" == "1" ]]; then
      echo "  recovery: GPU reset/reload failed; falling back to CPU because CUDA_FAILURE_CPU_FALLBACK=1"
      switch_training_to_cpu
    else
      CUDA_DRIVER_POISONED=1
      echo "  recovery: GPU reset/reload failed; preserving checkpoint and not relaunching CUDA" >&2
      break
    fi

    if [[ "$CUDA_FAILURE_CPU_FALLBACK" == "1" && "$USE_CUDA" != "0" ]]; then
      switch_training_to_cpu
    fi

    TRAIN_GENERATIONS_REMAINING=$((TRAIN_GENERATIONS_REMAINING - RECOVERED_GEN))
    TRAIN_RESUME_BUNDLE="$NEXT_DIR/self_play_models.json"
    if (( RECOVERED_POPULATION > 0 && RECOVERED_MIN_NODES > 0 && RECOVERED_MAX_NODES >= RECOVERED_MIN_NODES )); then
      TRAIN_POPULATION_SIZE="$RECOVERED_POPULATION"
      TRAIN_MIN_NODES="$RECOVERED_MIN_NODES"
      TRAIN_MAX_NODES="$RECOVERED_MAX_NODES"
    fi
    TRAIN_ATTEMPT=$((TRAIN_ATTEMPT + 1))
    echo "  recovery: resuming remaining $TRAIN_GENERATIONS_REMAINING generation(s)"
    echo
  done

  if (( TRAIN_STATUS != 0 )); then
    echo
    echo "Training warning"
    echo "  status: $TRAIN_STATUS"
    if [[ "$RECOVER_ON_TRAIN_FAILURE" != "1" ]]; then
      echo "Recovery disabled (RECOVER_ON_TRAIN_FAILURE=$RECOVER_ON_TRAIN_FAILURE); aborting." >&2
      exit "$TRAIN_STATUS"
    fi

    if [[ -f "$NEXT_DIR/self_play_models.json" && -f "$NEXT_DIR/searcher_model.json" && -f "$NEXT_DIR/best_evader_model.json" ]]; then
      RECOVERED_GEN="$(python3 - <<PYEOF || true
import json

try:
    with open("$NEXT_DIR/training_history.json") as fh:
        history = json.load(fh)
    print(history[-1].get("generation", "unknown"))
except Exception:
    print("unknown")
PYEOF
)"
      echo "Recovery checkpoint"
      echo "  directory:  $NEXT_DIR"
      echo "  generation: $RECOVERED_GEN"
      if [[ "$AUTO_GPU_RECOVER" == "1" && "$USE_CUDA" != "0" ]]; then
        if recover_gpu_if_enabled; then
          :
        else
          echo "WARNING: GPU recovery did not complete; the next CUDA health check will fail fast unless the driver is recovered manually." >&2
        fi
      fi
    else
      echo "ERROR: training failed before a usable recovery checkpoint was written in $NEXT_DIR" >&2
      exit "$TRAIN_STATUS"
    fi
  fi

  CURRICULUM_UPDATE="$(python3 - <<PYEOF || true
import json

history_path = "$NEXT_DIR/training_history.json"
try:
    with open(history_path) as fh:
        history = json.load(fh)
    last = history[-1]
    population = last.get("population_size")
    min_nodes = last.get("min_nodes")
    max_nodes = last.get("max_nodes")
    if population is not None and min_nodes is not None and max_nodes is not None:
        print(f"{population} {min_nodes} {max_nodes}")
except Exception:
    pass
PYEOF
)"
  if [[ "$CURRICULUM_UPDATE" =~ ^[0-9]+[[:space:]]+[0-9]+[[:space:]]+[0-9]+$ ]]; then
    read -r NEXT_POPULATION_SIZE NEXT_MIN_NODES NEXT_MAX_NODES <<<"$CURRICULUM_UPDATE"
    if [[ "$NEXT_POPULATION_SIZE" != "$CURRENT_POPULATION_SIZE" || "$NEXT_MIN_NODES" != "$CURRENT_MIN_NODES" || "$NEXT_MAX_NODES" != "$CURRENT_MAX_NODES" ]]; then
      echo "Adaptive curriculum carried forward"
      echo "  population: $CURRENT_POPULATION_SIZE -> $NEXT_POPULATION_SIZE"
      echo "  nodes:      ${CURRENT_MIN_NODES}..${CURRENT_MAX_NODES} -> ${NEXT_MIN_NODES}..${NEXT_MAX_NODES}"
    fi
    CURRENT_POPULATION_SIZE="$NEXT_POPULATION_SIZE"
    CURRENT_MIN_NODES="$NEXT_MIN_NODES"
    CURRENT_MAX_NODES="$NEXT_MAX_NODES"
  fi

  echo
  echo "Promotion"
  # Promote best_pair (joint snapshot at evader peak) if available; fall back to
  # best_evader + final_searcher for checkpoints trained before this change.
  python3 - <<PYEOF
import json, os, sys

out_dir = "$NEXT_DIR"

def load_first(*names):
    errors = []
    for name in names:
        path = f"{out_dir}/{name}"
        try:
            with open(path) as fh:
                return json.load(fh), path
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    raise RuntimeError("; ".join(errors))

try:
    best_pair_path = f"{out_dir}/best_pair_models.json"
    if os.path.exists(best_pair_path):
        with open(best_pair_path) as fh:
            bundle = json.load(fh)
        print(f"  action:   joint snapshot at evader peak (best_pair_models.json)")
        print(f"  evader:   {best_pair_path}")
        print(f"  searcher: {best_pair_path}")
    else:
        best_ev, best_ev_path = load_first("best_evader_model.json", "evader_model.json")
        final_sr, final_sr_path = load_first("searcher_model.json", "best_searcher_model.json")
        bundle = {"evader": best_ev, "searcher": final_sr}
        print(f"  action:   best evader + final searcher (no best_pair_models.json found)")
        print(f"  evader:   {best_ev_path}")
        print(f"  searcher: {final_sr_path}")
    with open(f"{out_dir}/self_play_models.json", "w") as fh:
        json.dump(bundle, fh, indent=2)
    print(f"  bundle:   {out_dir}/self_play_models.json")
except Exception as e:
    print(f"  ERROR: Could not promote models: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF

  echo
  echo "Evaluation"
  echo "  bundle: $(basename "$NEXT_DIR") promoted bundle"
  echo "  seed:   $ITER_SEED"

  if [[ "$CUDA_DRIVER_POISONED" == "1" && "$USE_CUDA" != "0" && "$CUDA_FAILURE_CPU_FALLBACK" != "1" ]]; then
    echo "CUDA stop"
    echo "  status: driver context still poisoned after recovery failure"
    echo "  action: not launching evaluation or another CUDA training attempt"
    echo "  resume: $NEXT_DIR/self_play_models.json"
    exit "$TRAIN_STATUS"
  fi

  EVAL_STATUS=1
  for ((eval_try = 1; eval_try <= MAX_EVAL_RESTARTS; eval_try++)); do
    EVAL_LOG="$NEXT_DIR/evaluate_attempt${eval_try}.log"
    EVAL_HEALTH_LOG="$NEXT_DIR/cuda_health_evaluate_attempt${eval_try}.log"
    EVAL_RESOURCE_LOG="$RESOURCE_LOG_DIR/$(basename "$NEXT_DIR")_evaluate_attempt${eval_try}.csv"
    echo "  launch:"
    echo "    attempt: $eval_try/$MAX_EVAL_RESTARTS"
    echo "    backend: $([[ "$USE_CUDA" == "0" ]] && echo "cpu" || echo "cuda")"
    echo "    log:     $EVAL_LOG"

    if ensure_cuda_healthy "evaluate attempt $eval_try" "$EVAL_HEALTH_LOG" "$MAX_EVAL_RESTARTS"; then
      :
    else
      EVAL_STATUS=$?
      echo "Evaluation CUDA health failed"
      echo "  log: $EVAL_HEALTH_LOG"
      if [[ "$CUDA_FAILURE_CPU_FALLBACK" == "1" ]]; then
        switch_training_to_cpu
        continue
      fi
      CUDA_DRIVER_POISONED=1
      break
    fi

    set +e
    start_resource_logger "$EVAL_RESOURCE_LOG"
    run_train_binary evaluate \
      --model-bundle "$NEXT_DIR/self_play_models.json" \
      --episodes "$EVAL_EPISODES" \
      --seed $((ITER_SEED + 7777)) \
      --min-nodes "$CURRENT_MIN_NODES" \
      --max-nodes "$CURRENT_MAX_NODES" \
      "${ATTEMPT_FLAG_ARGS[@]}" 2>&1 | (trap '' INT; tee "$EVAL_LOG")
    EVAL_STATUS=${PIPESTATUS[0]}
    stop_resource_logger
    set -e

    if (( EVAL_STATUS == 0 )); then
      break
    fi

    if ! cuda_failure_in_log "$EVAL_LOG"; then
      break
    fi

    echo "Evaluation CUDA failure detected"
    echo "  log: $EVAL_LOG"

    if (( eval_try >= MAX_EVAL_RESTARTS )); then
      break
    fi

    if recover_gpu_if_enabled; then
      :
    elif [[ "$CUDA_FAILURE_CPU_FALLBACK" == "1" ]]; then
      echo "Evaluation recovery failed; falling back to CPU because CUDA_FAILURE_CPU_FALLBACK=1"
      switch_training_to_cpu
    else
      echo "Evaluation recovery failed; not relaunching CUDA." >&2
      break
    fi

    if [[ "$CUDA_FAILURE_CPU_FALLBACK" == "1" && "$USE_CUDA" != "0" ]]; then
      switch_training_to_cpu
    fi
  done

  if (( EVAL_STATUS != 0 )); then
    echo "WARNING: evaluation exited with status $EVAL_STATUS; checkpoint remains usable at $NEXT_DIR/self_play_models.json"
  fi

  if [[ "$CUDA_DRIVER_POISONED" == "1" && "$USE_CUDA" != "0" && "$CUDA_FAILURE_CPU_FALLBACK" != "1" ]]; then
    echo "CUDA stop"
    echo "  status: driver context still poisoned after evaluation recovery failure"
    echo "  action: not launching another CUDA training attempt"
    echo "  resume: $NEXT_DIR/self_play_models.json"
    exit "$EVAL_STATUS"
  fi

  echo
  echo "Iteration checkpoint"
  echo "  best evader: $NEXT_DIR/best_evader_model.json"

  CURRENT_CHECKPOINT="$NEXT_DIR"
  echo

  if [[ "$USER_STOP_REQUESTED" == "1" ]]; then
    echo "Graceful stop: iteration $iter complete. Not starting further iterations."
    break
  fi
done

echo "All iterations complete"
echo "  final checkpoint: $CURRENT_CHECKPOINT"
