#!/usr/bin/env bash
# Launch the interactive ShellGame tree visualizer with a trained model.
#
# Model resolution order (first match wins):
#   1. Explicit path:  ./run_visualizer.sh models/my_run/coagent_stage/self_play_models.json
#   2. $MODEL env var: MODEL=models/fast_runs/run_3/coagent_stage/self_play_models.json ./run_visualizer.sh
#   3. Auto-detect:    highest-numbered run under $MODEL_DIR (coagent preferred over static)
#   4. Core fallback:  highest *NNeV* dir under $CORE_MODEL_DIR, matching select_highest_core.sh
#
# Visualizer options (env vars):
#   NODES=25          tree size (default: 21)
#   DELAY_MS=400      ms between steps (default: 350)
#   GENERATION=uneven tree shape: balanced|uneven (default: uneven)
#   HIDE_SHELL=1      hide the shell position indicator
#   AUTO_RERUN=1      automatically restart after each episode
#   INSTANT_AUTO_RERUN=1  remove the post-episode auto-rerun pause entirely
#   SHELL_BEHAVIOR=adaptive  shell behavior: static|random|adaptive (default: adaptive)
#   SEARCHER=model    search controller: algorithm|model (default: model)
#   ALGORITHM=evasion-aware  used when SEARCHER=algorithm
#   MODEL_DIR=models/pipeline_runs   where to auto-detect pipeline runs from (default: pipeline_runs)
#   CORE_MODEL_DIR=models/core       where to auto-detect core models from (default: models/core)
#   USE_CUDA=0        build/run the visualizer CPU-only
#   AUTO_GPU_RECOVER=1  try recover_gpu.sh if CUDA init fails (default: 1)
#   GPU_RECOVERY_ALLOW_PROMPT=1  allow sudo password prompt during recovery
#   GPU_RECOVERY_SCRIPT=./recover_gpu.sh  override the no-reboot recovery helper
#   CUDA_RECOVERY_RETRIES=3  CUDA launch attempts before giving up
#   CUDA_HEALTH_CHECK=1  run supervised CUDA matmul + batched MLP stress probe first
#   GPU_SINGLE_SCORE_ROWS=4096  cap rows per single-model CUDA MLP call
#   CPU_FALLBACK=1    opt into CPU fallback after CUDA recovery retries fail

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# ── Model resolution ───────────────────────────────────────────────────────────

MODEL_DIR="${MODEL_DIR:-models/pipeline_runs}"
CORE_MODEL_DIR="${CORE_MODEL_DIR:-models/core}"
EXPLICIT_MODEL="${1:-${MODEL:-}}"

resolve_highest_core_model() {
  local target_dir="$CORE_MODEL_DIR"
  [[ -d "$target_dir" ]] || return 1

  # Same selection idea as select_highest_core.sh:
  # extract the highest number immediately before "ev", then choose a matching dir.
  local highest_ev
  highest_ev="$(
    find "$target_dir" -maxdepth 1 -type d -printf '%f\n' 2>/dev/null \
      | grep -oE '[0-9]+ev' \
      | sed 's/ev$//' \
      | sort -rn \
      | head -n 1
  )"
  [[ -n "$highest_ev" ]] || return 1

  local core_dir=""
  while IFS= read -r candidate; do
    if [[ -f "$candidate/self_play_models.json" ]]; then
      core_dir="$candidate"
      break
    fi
  done < <(find "$target_dir" -maxdepth 1 -type d -name "*${highest_ev}ev*" 2>/dev/null)

  [[ -n "$core_dir" ]] || return 1
  printf '%s/self_play_models.json\n' "$core_dir"
}

if [[ -n "$EXPLICIT_MODEL" ]]; then
  MODEL_BUNDLE="$EXPLICIT_MODEL"
else
  # Auto-detect: find the highest-numbered run, prefer coagent stage.
  best=""
  best_num=-1
  while IFS= read -r candidate; do
    dir="$(dirname "$candidate")"
    run_dir="$(dirname "$dir")"
    run_name="$(basename "$run_dir")"
    num="${run_name##*_}"
    if [[ "$num" =~ ^[0-9]+$ ]] && (( num > best_num )); then
      best_num=$num
      best="$candidate"
    fi
  done < <(find "$MODEL_DIR" -path "*/coagent_stage/self_play_models.json" 2>/dev/null | sort)

  if [[ -z "$best" ]]; then
    # Fall back to static stage if no coagent model exists yet.
    while IFS= read -r candidate; do
      dir="$(dirname "$candidate")"
      run_dir="$(dirname "$dir")"
      run_name="$(basename "$run_dir")"
      num="${run_name##*_}"
      if [[ "$num" =~ ^[0-9]+$ ]] && (( num > best_num )); then
        best_num=$num
        best="$candidate"
      fi
    done < <(find "$MODEL_DIR" -path "*/static_stage/self_play_models.json" 2>/dev/null | sort)
  fi

  if [[ -z "$best" ]]; then
    best="$(resolve_highest_core_model || true)"
  fi

  if [[ -z "$best" ]]; then
    echo "ERROR: no model found under $MODEL_DIR or $CORE_MODEL_DIR" >&2
    echo "Train first, or pass a model path as the first argument." >&2
    exit 1
  fi

  MODEL_BUNDLE="$best"
fi

if [[ ! -f "$MODEL_BUNDLE" ]]; then
  echo "ERROR: model not found: $MODEL_BUNDLE" >&2
  exit 1
fi

# ── Visualizer parameters ──────────────────────────────────────────────────────

NODES="${NODES:-21}"
DELAY_MS="${DELAY_MS:-350}"
GENERATION="${GENERATION:-uneven}"
if [[ -n "${SHELL_BEHAVIOR:-}" ]]; then
  :
elif [[ "${SHELL:-}" =~ ^(static|random|adaptive)$ ]]; then
  SHELL_BEHAVIOR="$SHELL"
else
  SHELL_BEHAVIOR="adaptive"
fi
SEARCHER="${SEARCHER:-model}"
ALGORITHM="${ALGORITHM:-evasion-aware}"
HIDE_SHELL="${HIDE_SHELL:-0}"
AUTO_RERUN="${AUTO_RERUN:-0}"
INSTANT_AUTO_RERUN="${INSTANT_AUTO_RERUN:-0}"
MAX_ATTEMPTS_FACTOR="${MAX_ATTEMPTS_FACTOR:-2}"
GPU_SINGLE_SCORE_ROWS="${GPU_SINGLE_SCORE_ROWS:-4096}"
USE_CUDA="${USE_CUDA:-1}"
AUTO_GPU_RECOVER="${AUTO_GPU_RECOVER:-1}"
CPU_FALLBACK="${CPU_FALLBACK:-0}"
GPU_RECOVERY_ALLOW_PROMPT="${GPU_RECOVERY_ALLOW_PROMPT:-1}"
GPU_RECOVERY_SCRIPT="${GPU_RECOVERY_SCRIPT:-$(pwd)/recover_gpu.sh}"
CUDA_RECOVERY_RETRIES="${CUDA_RECOVERY_RETRIES:-3}"
CUDA_HEALTH_CHECK="${CUDA_HEALTH_CHECK:-1}"
CUDA_HEALTH_BIN="$(pwd)/target/release/cuda_health"
export GPU_RECOVERY_ALLOW_PROMPT
export GPU_SINGLE_SCORE_ROWS

CARGO_FEATURE_ARGS=()
if [[ "$USE_CUDA" == "0" ]]; then
  CARGO_FEATURE_ARGS=("--no-default-features")
else
  if [[ -z "${NVCC_CCBIN:-}" ]] && command -v gcc-13 &>/dev/null; then
    export NVCC_CCBIN=/usr/bin/gcc-13
  fi
fi

# ── Build ──────────────────────────────────────────────────────────────────────

echo "Model:    $MODEL_BUNDLE"
echo "Nodes:    $NODES  Generation: $GENERATION  Shell: $SHELL_BEHAVIOR  Searcher: $SEARCHER"
echo "Rerun:    auto=$AUTO_RERUN instant=$INSTANT_AUTO_RERUN"
echo "CUDA:     use=$USE_CUDA recover=$AUTO_GPU_RECOVER fallback_cpu=$CPU_FALLBACK"
echo "GPU rows: single_model=$GPU_SINGLE_SCORE_ROWS"
echo "Recovery: $GPU_RECOVERY_SCRIPT"
echo "Sudo:     gpu_recovery_prompt=$GPU_RECOVERY_ALLOW_PROMPT"
echo "Retries:  cuda=$CUDA_RECOVERY_RETRIES health_check=$CUDA_HEALTH_CHECK"

build_visualizer_binary() {
  cargo build --release "$@" --bin tree_visualizer 2>&1 | grep -v "^$" | grep -v "Compiling candle-kernels"
}

build_visualizer_binary "${CARGO_FEATURE_ARGS[@]}"
if [[ "$USE_CUDA" != "0" && "$CUDA_HEALTH_CHECK" == "1" ]]; then
  cargo build --release --bin cuda_health 2>&1 | grep -v "^$" | grep -v "Compiling candle-kernels"
fi
CPU_BUILD_READY=0
if [[ "$USE_CUDA" == "0" ]]; then
  CPU_BUILD_READY=1
fi

# ── Extra flags ────────────────────────────────────────────────────────────────

EXTRA_FLAGS=()
[[ "$HIDE_SHELL" == "1" ]] && EXTRA_FLAGS+=(--hide-shell)
[[ "$AUTO_RERUN" == "1" ]] && EXTRA_FLAGS+=(--auto-rerun)
[[ "$INSTANT_AUTO_RERUN" == "1" ]] && EXTRA_FLAGS+=(--instant-auto-rerun)
[[ -n "${MAX_ATTEMPTS_CAP:-}"   ]] && EXTRA_FLAGS+=(--max-attempts-cap   "$MAX_ATTEMPTS_CAP")
[[ -n "${MAX_ATTEMPTS_RATIO:-}" ]] && EXTRA_FLAGS+=(--max-attempts-ratio "$MAX_ATTEMPTS_RATIO")

# ── Launch ─────────────────────────────────────────────────────────────────────

VISUALIZER_CMD=(
  ./target/release/tree_visualizer
  --nodes "$NODES"
  --generation "$GENERATION"
  --shell-behavior "$SHELL_BEHAVIOR"
  --search-controller "$SEARCHER"
  --search-algorithm "$ALGORITHM"
  --delay-ms "$DELAY_MS"
  --max-attempts-factor "$MAX_ATTEMPTS_FACTOR"
  --model-bundle "$MODEL_BUNDLE"
  "${EXTRA_FLAGS[@]}"
)

CUDA_LOG="$(mktemp /tmp/shellgame_visualizer_cuda.XXXXXX.log)"
CUDA_HEALTH_LOG="$(mktemp /tmp/shellgame_visualizer_cuda_health.XXXXXX.log)"
trap 'rm -f "$CUDA_LOG" "$CUDA_HEALTH_LOG"' EXIT

is_cuda_failure_log() {
  local log_path="$1"
  grep -Eqi 'CUDA_ERROR|DriverError\(CUDA|unspecified launch failure|cudarc|cuda_backend|CudaSlice|GPU .*failed|CUDA .*failed' "$log_path"
}

run_visualizer_once() {
  local require_cuda="$1"
  if [[ "$require_cuda" == "1" ]]; then
    CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}" REQUIRE_CUDA=1 "${VISUALIZER_CMD[@]}" 2>&1 | tee "$CUDA_LOG"
    return "${PIPESTATUS[0]}"
  fi

  "${VISUALIZER_CMD[@]}"
}

run_cuda_health_once() {
  CUDA_LAUNCH_BLOCKING=1 \
    REQUIRE_CUDA=1 \
    CUDA_HEALTH_SINGLE_ROWS="${CUDA_HEALTH_SINGLE_ROWS:-$GPU_SINGLE_SCORE_ROWS}" \
    "$CUDA_HEALTH_BIN" 2>&1 | tee "$CUDA_HEALTH_LOG"
  return "${PIPESTATUS[0]}"
}

recover_gpu_if_enabled() {
  if [[ "$AUTO_GPU_RECOVER" == "1" && -x "$GPU_RECOVERY_SCRIPT" ]]; then
    echo "Attempting no-reboot GPU recovery..."
    if "$GPU_RECOVERY_SCRIPT"; then
      echo "GPU recovery completed."
      return 0
    else
      echo "WARNING: GPU recovery script could not reset/reload the driver." >&2
      return 1
    fi
  elif [[ "$AUTO_GPU_RECOVER" == "1" ]]; then
    echo "WARNING: GPU recovery script is missing or not executable: $GPU_RECOVERY_SCRIPT" >&2
    return 1
  fi

  echo "WARNING: automatic GPU recovery is disabled; not retrying unsafe CUDA." >&2
  return 1
}

ensure_cuda_healthy() {
  if [[ "$CUDA_HEALTH_CHECK" != "1" ]]; then
    return 0
  fi

  for ((health_try = 1; health_try <= CUDA_RECOVERY_RETRIES; health_try++)); do
    echo
    echo "CUDA health probe $health_try/$CUDA_RECOVERY_RETRIES"
    : > "$CUDA_HEALTH_LOG"
    set +e
    run_cuda_health_once
    local health_status=$?
    set -e
    if (( health_status == 0 )); then
      return 0
    fi
    if ! is_cuda_failure_log "$CUDA_HEALTH_LOG"; then
      return "$health_status"
    fi
    if (( health_try >= CUDA_RECOVERY_RETRIES )); then
      return "$health_status"
    fi
    if ! recover_gpu_if_enabled; then
      echo "ERROR: GPU recovery failed; refusing to relaunch CUDA into a poisoned driver context." >&2
      return "$health_status"
    fi
  done
}

launch_cpu_visualizer() {
  echo
  echo "Launching visualizer with the CPU-only build."
  if [[ "${CPU_BUILD_READY:-0}" != "1" ]]; then
    build_visualizer_binary --no-default-features
  fi
  SHELLGAME_FORCE_CPU=1 exec "${VISUALIZER_CMD[@]}"
}

if [[ "$USE_CUDA" == "0" ]]; then
  launch_cpu_visualizer
fi

if ensure_cuda_healthy; then
  :
else
  VIS_STATUS=$?
  echo "ERROR: CUDA health probe failed before launching the visualizer." >&2
  if [[ "$CPU_FALLBACK" == "1" ]]; then
    echo "CPU_FALLBACK=1 is set; launching CPU-only visualizer." >&2
    launch_cpu_visualizer
  fi
  exit "$VIS_STATUS"
fi

VIS_STATUS=1
for ((cuda_try = 1; cuda_try <= CUDA_RECOVERY_RETRIES; cuda_try++)); do
  echo
  echo "CUDA visualizer launch attempt $cuda_try/$CUDA_RECOVERY_RETRIES"
  : > "$CUDA_LOG"
  set +e
  run_visualizer_once 1
  VIS_STATUS=$?
  set -e

  if (( VIS_STATUS == 0 )); then
    exit 0
  fi

  if ! is_cuda_failure_log "$CUDA_LOG"; then
    exit "$VIS_STATUS"
  fi

  echo
  echo "CUDA failure detected before/during visualizer launch."
  if (( cuda_try >= CUDA_RECOVERY_RETRIES )); then
    break
  fi

  if ! recover_gpu_if_enabled; then
    echo "ERROR: GPU recovery failed; refusing another CUDA visualizer launch." >&2
    break
  fi
  if ensure_cuda_healthy; then
    :
  else
    VIS_STATUS=$?
    echo "ERROR: CUDA health probe still fails after recovery." >&2
    break
  fi
done

if [[ "$CPU_FALLBACK" == "1" ]]; then
  echo
  echo "CUDA is still unsafe; rebuilding and relaunching visualizer CPU-only."
  echo "Set CPU_FALLBACK=0 to abort instead."
  launch_cpu_visualizer
fi

echo "ERROR: CUDA is still unsafe after $CUDA_RECOVERY_RETRIES attempt(s), and CPU_FALLBACK=0." >&2
exit "$VIS_STATUS"
