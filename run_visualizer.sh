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
#   SHELL_BEHAVIOR=adaptive  shell behavior: static|random|adaptive (default: adaptive)
#   SEARCHER=model    search controller: algorithm|model (default: model)
#   ALGORITHM=evasion-aware  used when SEARCHER=algorithm
#   MODEL_DIR=models/pipeline_runs   where to auto-detect pipeline runs from (default: pipeline_runs)
#   CORE_MODEL_DIR=models/core       where to auto-detect core models from (default: models/core)
#   USE_CUDA=0        build/run the visualizer CPU-only
#   AUTO_GPU_RECOVER=1  try recover_gpu.sh if CUDA init fails (default: 1)
#   CPU_FALLBACK=1    after failed recovery, launch with CPU fallback instead of aborting

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
MAX_ATTEMPTS_FACTOR="${MAX_ATTEMPTS_FACTOR:-2}"
USE_CUDA="${USE_CUDA:-1}"
AUTO_GPU_RECOVER="${AUTO_GPU_RECOVER:-1}"
CPU_FALLBACK="${CPU_FALLBACK:-1}"
GPU_RECOVERY_SCRIPT="${GPU_RECOVERY_SCRIPT:-$(pwd)/recover_gpu.sh}"

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
echo "CUDA:     use=$USE_CUDA recover=$AUTO_GPU_RECOVER fallback_cpu=$CPU_FALLBACK"
echo "Recovery: $GPU_RECOVERY_SCRIPT"

cargo build --release "${CARGO_FEATURE_ARGS[@]}" --bin tree_visualizer 2>&1 | grep -v "^$" | grep -v "Compiling candle-kernels"

# ── Extra flags ────────────────────────────────────────────────────────────────

EXTRA_FLAGS=()
[[ "$HIDE_SHELL" == "1" ]] && EXTRA_FLAGS+=(--hide-shell)
[[ "$AUTO_RERUN" == "1" ]] && EXTRA_FLAGS+=(--auto-rerun)
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
trap 'rm -f "$CUDA_LOG"' EXIT

run_visualizer_once() {
  local require_cuda="$1"
  if [[ "$require_cuda" == "1" ]]; then
    REQUIRE_CUDA=1 "${VISUALIZER_CMD[@]}" 2>&1 | tee "$CUDA_LOG"
    return "${PIPESTATUS[0]}"
  fi

  "${VISUALIZER_CMD[@]}"
}

if [[ "$USE_CUDA" == "0" ]]; then
  exec "${VISUALIZER_CMD[@]}"
fi

set +e
run_visualizer_once 1
VIS_STATUS=$?
set -e

if (( VIS_STATUS == 0 )); then
  exit 0
fi

if ! grep -qi "CUDA init failed" "$CUDA_LOG"; then
  exit "$VIS_STATUS"
fi

echo
echo "CUDA init failed before the visualizer window opened."
if [[ "$AUTO_GPU_RECOVER" == "1" && -x "$GPU_RECOVERY_SCRIPT" ]]; then
  echo "Attempting no-reboot GPU recovery..."
  if "$GPU_RECOVERY_SCRIPT"; then
    echo "GPU recovery completed; retrying visualizer with CUDA required."
    : > "$CUDA_LOG"
    set +e
    run_visualizer_once 1
    VIS_STATUS=$?
    set -e
    if (( VIS_STATUS == 0 )); then
      exit 0
    fi
    if ! grep -qi "CUDA init failed" "$CUDA_LOG"; then
      exit "$VIS_STATUS"
    fi
  else
    echo "WARNING: GPU recovery script could not reset/reload the driver." >&2
  fi
elif [[ "$AUTO_GPU_RECOVER" == "1" ]]; then
  echo "WARNING: GPU recovery script is missing or not executable: $GPU_RECOVERY_SCRIPT" >&2
fi

if [[ "$CPU_FALLBACK" == "1" ]]; then
  echo
  echo "CUDA is still unavailable; launching visualizer with CPU fallback."
  echo "Set CPU_FALLBACK=0 to abort instead."
  exec "${VISUALIZER_CMD[@]}"
fi

echo "ERROR: CUDA is still unavailable and CPU_FALLBACK=0." >&2
exit "$VIS_STATUS"
