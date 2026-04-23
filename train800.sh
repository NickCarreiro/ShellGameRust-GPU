#!/usr/bin/env bash
# v2 — 800-generation pipeline: 400 static then 400 coagent.
# Model: 14→1024→512→1 two-hidden-layer MLP.
#
# GPU is enabled by default (CUDA is a default Cargo feature).
# Disable only for intentional CPU comparison: USE_CUDA=0 ./train800.sh
#
# CUDA 12.4 requires gcc-13 as the NVCC host compiler (gcc-14 is rejected).
# Install once: sudo apt install gcc-13 g++-13
#
# Static stage MUST come first — it builds the evasion foundation.
# Coagent stage resumes from the static model and generalises against a learning searcher.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

SEED="${SEED:-1337}"
OUTPUT_DIR="${OUTPUT_DIR:-models}"
STATIC_DIR="$OUTPUT_DIR/static_stage"
COAGENT_DIR="$OUTPUT_DIR/coagent_stage"
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

mkdir -p "$STATIC_DIR" "$COAGENT_DIR"

echo
echo "=== v2 Static stage: 400 generations (seed=$SEED) ==="
cargo run --release "${CARGO_FEATURE_ARGS[@]}" --bin ml_self_play -- train \
  --training-mode static \
  --generations 400 \
  --population-size 20 \
  --episodes-per-eval 40 \
  --static-opponent-sample-count 7 \
  --min-nodes 11 \
  --max-nodes 31 \
  --max-attempts-factor 2 \
  --max-attempts-ratio 0.40 \
  --max-attempts-cap 12 \
  --hall-of-fame-size 8 \
  --hall-sample-count 2 \
  --mutation-scale 0.15 \
  --seed "$SEED" \
  --output-dir "$STATIC_DIR"

echo
echo "=== v2 Coagent stage: 400 generations (seed=$SEED, resuming from static) ==="
cargo run --release "${CARGO_FEATURE_ARGS[@]}" --bin ml_self_play -- train \
  --training-mode coagent \
  --generations 400 \
  --population-size 20 \
  --episodes-per-eval 40 \
  --min-nodes 11 \
  --max-nodes 31 \
  --max-attempts-factor 2 \
  --max-attempts-ratio 0.40 \
  --max-attempts-cap 12 \
  --hall-of-fame-size 8 \
  --hall-sample-count 2 \
  --mutation-scale 0.15 \
  --seed "$SEED" \
  --resume-from "$STATIC_DIR/self_play_models.json" \
  --output-dir "$COAGENT_DIR"

echo
echo "=== Evaluation (200 episodes) ==="
cargo run --release "${CARGO_FEATURE_ARGS[@]}" --bin ml_self_play -- evaluate \
  --model-bundle "$COAGENT_DIR/self_play_models.json" \
  --episodes 200 \
  --seed "$SEED" \
  --min-nodes 11 \
  --max-nodes 31 \
  --max-attempts-factor 2 \
  --max-attempts-ratio 0.40 \
  --max-attempts-cap 12

echo
echo "Final model: $COAGENT_DIR/self_play_models.json"
