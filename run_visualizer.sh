#!/usr/bin/env bash
# Launch the interactive ShellGame tree visualizer with a trained model.
#
# Model resolution order (first match wins):
#   1. Explicit path:  ./run_visualizer.sh models/my_run/coagent_stage/self_play_models.json
#   2. $MODEL env var: MODEL=models/fast_runs/run_3/coagent_stage/self_play_models.json ./run_visualizer.sh
#   3. Auto-detect:    highest-numbered run under $MODEL_DIR (coagent preferred over static)
#
# Visualizer options (env vars):
#   NODES=25          tree size (default: 21)
#   DELAY_MS=400      ms between steps (default: 350)
#   GENERATION=uneven tree shape: balanced|uneven (default: uneven)
#   HIDE_SHELL=1      hide the shell position indicator
#   AUTO_RERUN=1      automatically restart after each episode
#   SHELL=adaptive    shell behavior: static|random|adaptive (default: adaptive)
#   SEARCHER=model    search controller: algorithm|model (default: model)
#   ALGORITHM=evasion-aware  used when SEARCHER=algorithm
#   MODEL_DIR=models/pipeline_runs   where to auto-detect from (default: pipeline_runs)

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# ── Model resolution ───────────────────────────────────────────────────────────

MODEL_DIR="${MODEL_DIR:-models/pipeline_runs}"
EXPLICIT_MODEL="${1:-${MODEL:-}}"

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
    echo "ERROR: no model found under $MODEL_DIR" >&2
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
SHELL_BEHAVIOR="${SHELL:-adaptive}"
SEARCHER="${SEARCHER:-model}"
ALGORITHM="${ALGORITHM:-evasion-aware}"
HIDE_SHELL="${HIDE_SHELL:-0}"
AUTO_RERUN="${AUTO_RERUN:-0}"
MAX_ATTEMPTS_FACTOR="${MAX_ATTEMPTS_FACTOR:-2}"

# ── Build ──────────────────────────────────────────────────────────────────────

echo "Model:    $MODEL_BUNDLE"
echo "Nodes:    $NODES  Generation: $GENERATION  Shell: $SHELL_BEHAVIOR  Searcher: $SEARCHER"

cargo build --release --bin tree_visualizer 2>&1 | grep -v "^$" | grep -v "Compiling candle-kernels"

# ── Extra flags ────────────────────────────────────────────────────────────────

EXTRA_FLAGS=()
[[ "$HIDE_SHELL" == "1" ]] && EXTRA_FLAGS+=(--hide-shell)
[[ "$AUTO_RERUN" == "1" ]] && EXTRA_FLAGS+=(--auto-rerun)
[[ -n "${MAX_ATTEMPTS_CAP:-}"   ]] && EXTRA_FLAGS+=(--max-attempts-cap   "$MAX_ATTEMPTS_CAP")
[[ -n "${MAX_ATTEMPTS_RATIO:-}" ]] && EXTRA_FLAGS+=(--max-attempts-ratio "$MAX_ATTEMPTS_RATIO")

# ── Launch ─────────────────────────────────────────────────────────────────────

exec ./target/release/tree_visualizer \
  --nodes              "$NODES" \
  --generation         "$GENERATION" \
  --shell-behavior     "$SHELL_BEHAVIOR" \
  --search-controller  "$SEARCHER" \
  --search-algorithm   "$ALGORITHM" \
  --delay-ms           "$DELAY_MS" \
  --max-attempts-factor "$MAX_ATTEMPTS_FACTOR" \
  --model-bundle       "$MODEL_BUNDLE" \
  "${EXTRA_FLAGS[@]}"
