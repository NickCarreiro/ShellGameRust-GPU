#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${CORE_CLEAN_TARGET_DIR:-$ROOT_DIR/models/core}"
KEEP_FILE="${CORE_CLEAN_KEEP_FILE:-manifest.json}"
SCORE_SOURCE="${CORE_CLEAN_SCORE_SOURCE:-evaluate}" # evaluate|history

EVAL_EPISODES="${CORE_CLEAN_EPISODES:-100}"
EVAL_SEED="${CORE_CLEAN_SEED:-17776}"
MIN_NODES="${MIN_NODES:-30}"
MAX_NODES="${MAX_NODES:-50}"
MAX_ATTEMPTS_FACTOR="${MAX_ATTEMPTS_FACTOR:-4}"
MAX_ATTEMPTS_RATIO="${MAX_ATTEMPTS_RATIO:-0.40}"
MAX_ATTEMPTS_CAP="${MAX_ATTEMPTS_CAP:-}"
USE_CUDA="${USE_CUDA:-1}"

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
elif [[ -n "${1:-}" ]]; then
  echo "Usage: ./core_clean.sh [--force]" >&2
  echo "Env: CORE_CLEAN_SCORE_SOURCE=evaluate|history CORE_CLEAN_EPISODES=100 USE_CUDA=1" >&2
  exit 2
fi

if [[ ! -d "$TARGET_DIR" ]]; then
  echo "ERROR: core directory does not exist: $TARGET_DIR" >&2
  exit 1
fi

CARGO_FEATURE_ARGS=()
if [[ "$USE_CUDA" == "0" ]]; then
  CARGO_FEATURE_ARGS=(--no-default-features)
  export SHELLGAME_FORCE_CPU=1
elif [[ -z "${NVCC_CCBIN:-}" ]] && command -v gcc-13 >/dev/null 2>&1; then
  export NVCC_CCBIN=/usr/bin/gcc-13
fi

TRAIN_BIN="$ROOT_DIR/target/release/ml_self_play"

evaluate_evader_score() {
  local model_bundle="$1"
  local cmd=(
    "$TRAIN_BIN" evaluate
    --model-bundle "$model_bundle"
    --episodes "$EVAL_EPISODES"
    --seed "$EVAL_SEED"
    --min-nodes "$MIN_NODES"
    --max-nodes "$MAX_NODES"
    --max-attempts-factor "$MAX_ATTEMPTS_FACTOR"
  )
  [[ -n "$MAX_ATTEMPTS_RATIO" ]] && cmd+=(--max-attempts-ratio "$MAX_ATTEMPTS_RATIO")
  [[ -n "$MAX_ATTEMPTS_CAP" ]] && cmd+=(--max-attempts-cap "$MAX_ATTEMPTS_CAP")

  local output
  if ! output="$(REQUIRE_CUDA="$USE_CUDA" CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}" "${cmd[@]}" 2>&1)"; then
    echo "$output" >&2
    return 1
  fi

  awk '/^[[:space:]]*evader:/ { print $2; found=1 } END { if (!found) exit 1 }' <<<"$output"
}

history_evader_score() {
  local model_dir="$1"
  python3 - "$model_dir" <<'PY'
import json
import math
import re
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
history_path = model_dir / "training_history.json"
scores = []
if history_path.exists():
    with history_path.open() as fh:
        history = json.load(fh)
    for record in history:
        value = record.get("evader_score")
        if isinstance(value, (int, float)) and math.isfinite(value):
            scores.append(float(value))

if scores:
    print(max(scores))
    raise SystemExit(0)

match = re.search(r"([0-9]+)ev", model_dir.name)
if match:
    print(float(match.group(1)))
    raise SystemExit(0)

raise SystemExit(1)
PY
}

score_model_dir() {
  local model_dir="$1"
  if [[ "$SCORE_SOURCE" == "evaluate" ]]; then
    evaluate_evader_score "$model_dir/self_play_models.json"
  elif [[ "$SCORE_SOURCE" == "history" ]]; then
    history_evader_score "$model_dir"
  else
    echo "ERROR: CORE_CLEAN_SCORE_SOURCE must be evaluate or history, got: $SCORE_SOURCE" >&2
    return 2
  fi
}

score_is_better() {
  local candidate_score="$1"
  local candidate_mtime="$2"
  local best_score="$3"
  local best_mtime="$4"
  python3 - "$candidate_score" "$candidate_mtime" "$best_score" "$best_mtime" <<'PY'
import math
import sys

candidate = float(sys.argv[1])
candidate_mtime = int(sys.argv[2])
best_raw = sys.argv[3]
best = float(best_raw) if best_raw else -math.inf
best_mtime = int(sys.argv[4]) if sys.argv[4] else -1

eps = 1e-9
if candidate > best + eps:
    raise SystemExit(0)
if abs(candidate - best) <= eps and candidate_mtime > best_mtime:
    raise SystemExit(0)
raise SystemExit(1)
PY
}

mapfile -t MODEL_DIRS < <(
  find "$TARGET_DIR" -mindepth 1 -maxdepth 1 -type d -exec test -f '{}/self_play_models.json' ';' -print | sort
)

if (( ${#MODEL_DIRS[@]} == 0 )); then
  echo "ERROR: no core directories with self_play_models.json found under $TARGET_DIR" >&2
  exit 1
fi

if [[ "$SCORE_SOURCE" == "evaluate" ]]; then
  echo "Build"
  echo "  binary: $TRAIN_BIN"
  cargo build --release "${CARGO_FEATURE_ARGS[@]}" --bin ml_self_play
  echo
fi

best_dir=""
best_score=""
best_mtime=""

echo "--- Core Cleanup ---"
echo "Target:       $TARGET_DIR"
echo "Score source: $SCORE_SOURCE"
if [[ "$SCORE_SOURCE" == "evaluate" ]]; then
  echo "Evaluation:   episodes=$EVAL_EPISODES seed=$EVAL_SEED nodes=${MIN_NODES}..${MAX_NODES} attempts=factor=$MAX_ATTEMPTS_FACTOR ratio=${MAX_ATTEMPTS_RATIO:-none} cap=${MAX_ATTEMPTS_CAP:-none}"
fi
echo "Candidates:"

for model_dir in "${MODEL_DIRS[@]}"; do
  name="$(basename "$model_dir")"
  mtime="$(stat -c %Y "$model_dir" 2>/dev/null || echo 0)"
  echo "  scoring: $name" >&2
  score="$(score_model_dir "$model_dir")" || {
    echo "ERROR: could not determine evader score for $model_dir" >&2
    exit 1
  }
  printf '  %-72s evader=%10.3f\n' "$name" "$score"
  if score_is_better "$score" "$mtime" "$best_score" "$best_mtime"; then
    best_dir="$model_dir"
    best_score="$score"
    best_mtime="$mtime"
  fi
done

KEEP_DIR="$(basename "$best_dir")"

echo "--------------------"
printf 'KEEPING:     %s (measured evader=%.3f)\n' "$KEEP_DIR" "$best_score"
echo "KEEPING:     $KEEP_FILE"
if (( FORCE == 0 )); then
  echo "Mode:        dry run"
else
  echo "Mode:        destructive"
fi
echo "--------------------"

for item in "$TARGET_DIR"/*; do
  basename_item="$(basename "$item")"
  if [[ "$basename_item" == "$KEEP_DIR" || "$basename_item" == "$KEEP_FILE" ]]; then
    continue
  fi

  if (( FORCE == 1 )); then
    echo "Deleting: $basename_item"
    rm -rf "$item"
  else
    echo "[DRY RUN] Would delete: $basename_item"
  fi
done

if (( FORCE == 0 )); then
  echo
  echo "!!! This was a DRY RUN. No files were deleted. !!!"
  echo "Run: ./core_clean.sh --force  to execute cleanup."
fi
