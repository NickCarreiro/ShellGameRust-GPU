#!/usr/bin/env bash
# Fast iteration script for testing training design changes.
#
# Tuned to saturate the GPU while keeping wall-clock time low:
#   - Population 40 vs 20       (bigger GPU batches; GPU handles extra candidates in parallel,
#                                 CPU tree-sim cost barely changes since it's shared per episode)
#   - Episodes 24 vs 40         (noisier fitness, but trends still visible)
#   - Static 40 gens vs 100     (saturates at ~20 anyway)
#   - Coagent 100 gens vs 150   (enough to see convergence trend)
#   - Nodes 11-25 vs 11-31      (min=11 matches production — below this evader has no hiding room)
#   - Hall of fame 4 vs 8
#   - Static opponents 3 vs 7 per generation
#
# GPU note: with population=10 the GPU sits at ~35% util / 2% VRAM.
# Population=40 pushes batches large enough to keep the GPU busy without
# proportionally increasing wall-clock time (stacked matmul scales well).
#
# Use the main train_static_then_coagent.sh for production runs.
# Use this script to validate a design change before committing to a full run.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STATIC_GENERATIONS="${STATIC_GENERATIONS:-40}"
COAGENT_GENERATIONS="${COAGENT_GENERATIONS:-100}"
POPULATION_SIZE="${POPULATION_SIZE:-40}"
EPISODES_PER_EVAL="${EPISODES_PER_EVAL:-24}"
MIN_NODES="${MIN_NODES:-11}"
MAX_NODES="${MAX_NODES:-25}"
MAX_ATTEMPTS_FACTOR="${MAX_ATTEMPTS_FACTOR:-2}"
MAX_ATTEMPTS_RATIO="${MAX_ATTEMPTS_RATIO:-0.40}"
MAX_ATTEMPTS_CAP="${MAX_ATTEMPTS_CAP:-10}"
STATIC_OPPONENT_SAMPLE_COUNT="${STATIC_OPPONENT_SAMPLE_COUNT:-3}"
HALL_OF_FAME_SIZE="${HALL_OF_FAME_SIZE:-4}"
HALL_SAMPLE_COUNT="${HALL_SAMPLE_COUNT:-1}"
MUTATION_SCALE="${MUTATION_SCALE:-0.18}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
BASE_SEED="${SEED:-1337}"
ITERATIONS="${1:-1}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-$ROOT_DIR/models/fast_runs}"
OVERWRITE_RUNS="${OVERWRITE_RUNS:-0}"
USE_CUDA="${USE_CUDA:-1}"

CARGO_FEATURE_ARGS=()
if [[ "$USE_CUDA" == "0" ]]; then
  CARGO_FEATURE_ARGS=("--no-default-features")
  echo "Cargo accelerator: CPU-only build (--no-default-features)"
else
  echo "Cargo accelerator: default (CUDA enabled)"
  if [[ -z "${NVCC_CCBIN:-}" ]]; then
    if command -v gcc-13 &>/dev/null; then
      export NVCC_CCBIN=/usr/bin/gcc-13
      echo "CUDA build: NVCC_CCBIN=$NVCC_CCBIN"
    else
      echo "WARNING: gcc-13 not found — CUDA build may fail."
      echo "  Fix: sudo apt install gcc-13 g++-13"
    fi
  fi
fi

mkdir -p "$BASE_OUTPUT_DIR"

echo
echo "Fast-iteration config:"
echo "  iterations=$ITERATIONS"
echo "  static_generations=$STATIC_GENERATIONS  (production: 100)"
echo "  coagent_generations=$COAGENT_GENERATIONS  (production: 150)"
echo "  population_size=$POPULATION_SIZE  (production: 20 — larger pop saturates GPU without proportional slowdown)"
echo "  episodes_per_eval=$EPISODES_PER_EVAL  (production: 40)"
echo "  nodes=${MIN_NODES}..${MAX_NODES}  (production: 11..31)"
echo "  static_opponent_sample_count=$STATIC_OPPONENT_SAMPLE_COUNT  (production: 7)"
echo "  hall_of_fame_size=$HALL_OF_FAME_SIZE  (production: 8)"
echo "  base_output_dir=$BASE_OUTPUT_DIR"

for ((iteration = 1; iteration <= ITERATIONS; iteration++)); do
  ITER_SEED=$((BASE_SEED + (iteration - 1) * 997))

  RUN_DIR="$BASE_OUTPUT_DIR/run_${iteration}"
  STATIC_DIR="$RUN_DIR/static_stage"
  COAGENT_DIR="$RUN_DIR/coagent_stage"

  if [[ -e "$RUN_DIR" ]]; then
    if [[ "$OVERWRITE_RUNS" == "1" ]]; then
      rm -rf "$RUN_DIR"
    else
      echo "ERROR: $RUN_DIR already exists." >&2
      echo "Use OVERWRITE_RUNS=1 or a fresh BASE_OUTPUT_DIR." >&2
      exit 1
    fi
  fi

  mkdir -p "$STATIC_DIR" "$COAGENT_DIR"

  echo
  echo "=== Iteration $iteration / $ITERATIONS: static stage (seed=$ITER_SEED) ==="
  cargo run --release "${CARGO_FEATURE_ARGS[@]}" --bin ml_self_play -- train \
    --training-mode static \
    --generations "$STATIC_GENERATIONS" \
    --population-size "$POPULATION_SIZE" \
    --episodes-per-eval "$EPISODES_PER_EVAL" \
    --static-opponent-sample-count "$STATIC_OPPONENT_SAMPLE_COUNT" \
    --min-nodes "$MIN_NODES" \
    --max-nodes "$MAX_NODES" \
    --max-attempts-factor "$MAX_ATTEMPTS_FACTOR" \
    --max-attempts-ratio "$MAX_ATTEMPTS_RATIO" \
    --max-attempts-cap "$MAX_ATTEMPTS_CAP" \
    --hall-of-fame-size "$HALL_OF_FAME_SIZE" \
    --hall-sample-count "$HALL_SAMPLE_COUNT" \
    --mutation-scale "$MUTATION_SCALE" \
    --seed "$ITER_SEED" \
    --output-dir "$STATIC_DIR"

  echo
  echo "=== Iteration $iteration / $ITERATIONS: coagent stage (seed=$ITER_SEED) ==="
  cargo run --release "${CARGO_FEATURE_ARGS[@]}" --bin ml_self_play -- train \
    --training-mode coagent \
    --generations "$COAGENT_GENERATIONS" \
    --population-size "$POPULATION_SIZE" \
    --episodes-per-eval "$EPISODES_PER_EVAL" \
    --min-nodes "$MIN_NODES" \
    --max-nodes "$MAX_NODES" \
    --max-attempts-factor "$MAX_ATTEMPTS_FACTOR" \
    --max-attempts-ratio "$MAX_ATTEMPTS_RATIO" \
    --max-attempts-cap "$MAX_ATTEMPTS_CAP" \
    --hall-of-fame-size "$HALL_OF_FAME_SIZE" \
    --hall-sample-count "$HALL_SAMPLE_COUNT" \
    --static-opponent-sample-count "$STATIC_OPPONENT_SAMPLE_COUNT" \
    --mutation-scale "$MUTATION_SCALE" \
    --seed "$ITER_SEED" \
    --resume-from "$STATIC_DIR/self_play_models.json" \
    --output-dir "$COAGENT_DIR"

  echo
  echo "=== Iteration $iteration / $ITERATIONS: evaluation (seed=$ITER_SEED) ==="
  cargo run --release "${CARGO_FEATURE_ARGS[@]}" --bin ml_self_play -- evaluate \
    --model-bundle "$COAGENT_DIR/self_play_models.json" \
    --episodes "$EVAL_EPISODES" \
    --seed "$ITER_SEED" \
    --min-nodes "$MIN_NODES" \
    --max-nodes "$MAX_NODES" \
    --max-attempts-factor "$MAX_ATTEMPTS_FACTOR" \
    --max-attempts-ratio "$MAX_ATTEMPTS_RATIO" \
    --max-attempts-cap "$MAX_ATTEMPTS_CAP"
done
