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

handle_interrupt() {
  if [[ "${TRAIN_CHILD_RUNNING:-0}" == "1" ]]; then
    # The trainer has its own Ctrl+C handler and will stop at a checkpoint boundary.
    return 0
  fi
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
        if [[ "$dir_name" =~ ([0-9]+)ev ]]; then
          ev_score="${BASH_REMATCH[1]}"
          mtime="$(stat -c %Y "$candidate" 2>/dev/null || echo 0)"
          if (( ev_score > best_ev || (ev_score == best_ev && mtime > best_mtime) )); then
            best_ev=$ev_score
            best_mtime=$mtime
            best_dir="$candidate"
          fi
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

GENERATIONS="${GENERATIONS:-20}"
POPULATION_SIZE="${POPULATION_SIZE:-50}"
EPISODES_PER_EVAL="${EPISODES_PER_EVAL:-45}"
MIN_NODES="${MIN_NODES:-30}"
MAX_NODES="${MAX_NODES:-50}"
MAX_ATTEMPTS_FACTOR="${MAX_ATTEMPTS_FACTOR:-4}"
MAX_ATTEMPTS_RATIO="${MAX_ATTEMPTS_RATIO:-0.40}"
# Leave cap unset by default; ratio-based budget scales naturally with node count.
MAX_ATTEMPTS_CAP="${MAX_ATTEMPTS_CAP:-}"
GPU_SCORE_BATCH_CELLS="${GPU_SCORE_BATCH_CELLS:-25600}"
GPU_SCORE_BATCH_ROWS="${GPU_SCORE_BATCH_ROWS:-}"
HALL_OF_FAME_SIZE="${HALL_OF_FAME_SIZE:-8}"
HALL_SAMPLE_COUNT="${HALL_SAMPLE_COUNT:-2}"
STATIC_OPPONENT_SAMPLE_COUNT="${STATIC_OPPONENT_SAMPLE_COUNT:-4}"
MUTATION_SCALE="${MUTATION_SCALE:-0.18}"
# Default patience tracks generation count so early-stopping is meaningful.
PATIENCE="${PATIENCE:-$GENERATIONS}"
# Adaptive curriculum: when evader score stagnates, widen the tree range and
# spend a larger ES population before patience gives up on the run.
STAGNATION_GROW_AFTER="${STAGNATION_GROW_AFTER:-5}"
STAGNATION_NODE_STEP="${STAGNATION_NODE_STEP:-5}"
STAGNATION_POPULATION_STEP="${STAGNATION_POPULATION_STEP:-25}"
STAGNATION_MAX_NODES_CAP="${STAGNATION_MAX_NODES_CAP:-}"
STAGNATION_POPULATION_CAP="${STAGNATION_POPULATION_CAP:-}"
BASE_SEED="${SEED:-9999}"
EVAL_EPISODES="${EVAL_EPISODES:-100}"
USE_CUDA="${USE_CUDA:-1}"
RECOVER_ON_TRAIN_FAILURE="${RECOVER_ON_TRAIN_FAILURE:-1}"
AUTO_GPU_RECOVER="${AUTO_GPU_RECOVER:-1}"
GPU_RECOVERY_SCRIPT="${GPU_RECOVERY_SCRIPT:-$ROOT_DIR/recover_gpu.sh}"

CARGO_FEATURE_ARGS=()
if [[ "$USE_CUDA" == "0" ]]; then
  CARGO_FEATURE_ARGS=("--no-default-features")
else
  if [[ -z "${NVCC_CCBIN:-}" ]] && command -v gcc-13 &>/dev/null; then
    export NVCC_CCBIN=/usr/bin/gcc-13
  fi
fi

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
echo "  attempt budget:    factor=$MAX_ATTEMPTS_FACTOR ratio=${MAX_ATTEMPTS_RATIO:-none} cap=${MAX_ATTEMPTS_CAP:-none}"
echo "  gpu score batch:   cells=$GPU_SCORE_BATCH_CELLS rows=${GPU_SCORE_BATCH_ROWS:-auto}"
if [[ ${#ADAPTIVE_GROWTH_ARGS[@]} -gt 0 ]]; then
  echo "  adaptive growth:   after=$STAGNATION_GROW_AFTER node_step=$STAGNATION_NODE_STEP population_step=$STAGNATION_POPULATION_STEP"
  echo "                     node_cap=${STAGNATION_MAX_NODES_CAP:-none} population_cap=${STAGNATION_POPULATION_CAP:-none}"
else
  echo "  adaptive growth:   disabled"
fi
echo "  train recovery:    $RECOVER_ON_TRAIN_FAILURE"
echo "  gpu recovery:      auto=$AUTO_GPU_RECOVER"
echo "                     script=$GPU_RECOVERY_SCRIPT"
echo

echo "Build"
echo "  binary: $TRAIN_BIN"
cargo build --release "${CARGO_FEATURE_ARGS[@]}" --bin ml_self_play
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
  set +e
  TRAIN_CHILD_RUNNING=1
  "$TRAIN_BIN" train \
    --training-mode coagent \
    --generations "$GENERATIONS" \
    --population-size "$CURRENT_POPULATION_SIZE" \
    --episodes-per-eval "$EPISODES_PER_EVAL" \
    --min-nodes "$CURRENT_MIN_NODES" \
    --max-nodes "$CURRENT_MAX_NODES" \
    "${ATTEMPT_FLAG_ARGS[@]}" \
    --hall-of-fame-size "$HALL_OF_FAME_SIZE" \
    --hall-sample-count "$HALL_SAMPLE_COUNT" \
    --static-opponent-sample-count "$STATIC_OPPONENT_SAMPLE_COUNT" \
    --mutation-scale "$MUTATION_SCALE" \
    --patience "$PATIENCE" \
    "${ADAPTIVE_GROWTH_ARGS[@]}" \
    --seed "$ITER_SEED" \
    --resume-from "$CURRENT_CHECKPOINT/self_play_models.json" \
    --output-dir "$NEXT_DIR"
  TRAIN_STATUS=$?
  TRAIN_CHILD_RUNNING=0
  set -e

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
        if [[ -x "$GPU_RECOVERY_SCRIPT" ]]; then
          echo "GPU recovery"
          echo "  action: attempting no-reboot recovery before evaluation/next iteration"
          if "$GPU_RECOVERY_SCRIPT"; then
            echo "  status: completed"
          else
            echo "WARNING: GPU recovery script could not reset/reload the driver. Check the manual commands above if CUDA fails again." >&2
          fi
        else
          echo "WARNING: GPU recovery script is missing or not executable: $GPU_RECOVERY_SCRIPT" >&2
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
  echo "  action: best evader + final searcher -> next-iteration bundle"
  # Promote best_evader into next checkpoint's bundle
  python3 - <<PYEOF
import json
import sys

out_dir  = "$NEXT_DIR"

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
    best_ev, best_ev_path = load_first("best_evader_model.json", "evader_model.json")
    final_sr, final_sr_path = load_first("searcher_model.json", "best_searcher_model.json")
    bundle = {"evader": best_ev, "searcher": final_sr}
    with open(f"{out_dir}/self_play_models.json", "w") as fh:
        json.dump(bundle, fh, indent=2)
    print(f"  evader:  {best_ev_path}")
    print(f"  searcher: {final_sr_path}")
    print(f"  bundle:  {out_dir}/self_play_models.json")
except Exception as e:
    print(f"  ERROR: Could not promote models: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF

  echo
  echo "Evaluation"
  echo "  bundle: $(basename "$NEXT_DIR") promoted bundle"
  echo "  seed:   $ITER_SEED"
  set +e
  "$TRAIN_BIN" evaluate \
    --model-bundle "$NEXT_DIR/self_play_models.json" \
    --episodes "$EVAL_EPISODES" \
    --seed $((ITER_SEED + 7777)) \
    --min-nodes "$CURRENT_MIN_NODES" \
    --max-nodes "$CURRENT_MAX_NODES" \
    "${ATTEMPT_FLAG_ARGS[@]}"
  EVAL_STATUS=$?
  set -e
  if (( EVAL_STATUS != 0 )); then
    echo "WARNING: evaluation exited with status $EVAL_STATUS; checkpoint remains usable at $NEXT_DIR/self_play_models.json"
  fi

  echo
  echo "Iteration checkpoint"
  echo "  best evader: $NEXT_DIR/best_evader_model.json"

  CURRENT_CHECKPOINT="$NEXT_DIR"
  echo
done

echo "All iterations complete"
echo "  final checkpoint: $CURRENT_CHECKPOINT"
