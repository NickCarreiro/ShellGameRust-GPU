#!/usr/bin/env bash
# Multi-run pipeline: N iterations of static-then-coagent training.
# Each iteration uses a distinct seed so results are independent experiments.
#
# GPU is enabled by default (CUDA is a default Cargo feature).
# Disable only for intentional CPU comparison: USE_CUDA=0 ./train_static_then_coagent.sh
#
# CUDA 12.4 requires gcc-13 as the NVCC host compiler (gcc-14 is rejected).
# Install once: sudo apt install gcc-13 g++-13

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STATIC_GENERATIONS="${STATIC_GENERATIONS:-100}"
COAGENT_GENERATIONS="${COAGENT_GENERATIONS:-150}"
POPULATION_SIZE="${POPULATION_SIZE:-20}"
EPISODES_PER_EVAL="${EPISODES_PER_EVAL:-40}"
MIN_NODES="${MIN_NODES:-11}"
MAX_NODES="${MAX_NODES:-31}"
MAX_ATTEMPTS_FACTOR="${MAX_ATTEMPTS_FACTOR:-2}"
MAX_ATTEMPTS_RATIO="${MAX_ATTEMPTS_RATIO:-0.40}"
MAX_ATTEMPTS_CAP="${MAX_ATTEMPTS_CAP:-12}"
STATIC_OPPONENT_SAMPLE_COUNT="${STATIC_OPPONENT_SAMPLE_COUNT:-7}"
HALL_OF_FAME_SIZE="${HALL_OF_FAME_SIZE:-8}"
HALL_SAMPLE_COUNT="${HALL_SAMPLE_COUNT:-2}"
MUTATION_SCALE="${MUTATION_SCALE:-0.15}"
EVAL_EPISODES="${EVAL_EPISODES:-100}"
BASE_SEED="${SEED:-1337}"
ITERATIONS="${1:-1}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-$ROOT_DIR/models/pipeline_runs}"
OVERWRITE_RUNS="${OVERWRITE_RUNS:-0}"
USE_CUDA="${USE_CUDA:-1}"

CARGO_FEATURE_ARGS=()
if [[ "$USE_CUDA" == "0" ]]; then
  CARGO_FEATURE_ARGS=("--no-default-features")
  echo "Cargo accelerator: CPU-only build (--no-default-features)"
else
  echo "Cargo accelerator: default (CUDA enabled)"
  # CUDA 12.4 rejects gcc-14; wire in gcc-13 as the NVCC host compiler.
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
echo "Pipeline config:"
echo "  iterations=$ITERATIONS"
echo "  static_generations=$STATIC_GENERATIONS"
echo "  coagent_generations=$COAGENT_GENERATIONS"
echo "  population_size=$POPULATION_SIZE"
echo "  episodes_per_eval=$EPISODES_PER_EVAL"
echo "  nodes=${MIN_NODES}..${MAX_NODES}"
echo "  attempts: factor=$MAX_ATTEMPTS_FACTOR ratio=$MAX_ATTEMPTS_RATIO cap=$MAX_ATTEMPTS_CAP"
echo "  static_opponent_sample_count=$STATIC_OPPONENT_SAMPLE_COUNT"
echo "  hall_of_fame_size=$HALL_OF_FAME_SIZE"
echo "  hall_sample_count=$HALL_SAMPLE_COUNT"
echo "  mutation_scale=$MUTATION_SCALE"
echo "  eval_episodes=$EVAL_EPISODES"
echo "  base_seed=$BASE_SEED"
echo "  base_output_dir=$BASE_OUTPUT_DIR"
echo "  overwrite_runs=$OVERWRITE_RUNS"
echo "  cargo_features=${CARGO_FEATURE_ARGS[*]:-default (CUDA)}"

for ((iteration = 1; iteration <= ITERATIONS; iteration++)); do
  # Each run gets a distinct seed so repeated runs produce genuinely different experiments.
  ITER_SEED=$((BASE_SEED + (iteration - 1) * 997))

  RUN_DIR="$BASE_OUTPUT_DIR/run_${iteration}"
  STATIC_DIR="$RUN_DIR/static_stage"
  COAGENT_DIR="$RUN_DIR/coagent_stage"

  if [[ -e "$RUN_DIR" ]]; then
    if [[ "$OVERWRITE_RUNS" == "1" ]]; then
      rm -rf "$RUN_DIR"
    else
      echo "ERROR: $RUN_DIR already exists." >&2
      echo "Use a fresh BASE_OUTPUT_DIR or set OVERWRITE_RUNS=1 to replace existing run output." >&2
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
