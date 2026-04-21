#!/bin/bash
# LoRA fine-tuning of Qwen3.5-0.8B-Base on MathQA → SELPH s-expressions
#
# Usage: ./train.sh [--iters N] [--batch-size N] [--learning-rate LR]
#
# Default: 1000 iters, batch 4, lr 1e-5, mask prompt (only train on completions)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL="Qwen/Qwen3.5-0.8B-Base"
DATA_DIR="$SCRIPT_DIR/training_data"
ADAPTER_DIR="$SCRIPT_DIR/adapters"

# Defaults
ITERS=1000
BATCH_SIZE=4
LR=1e-5
NUM_LAYERS=16
EXTRA_ARGS=()

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --iters) ITERS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --learning-rate) LR="$2"; shift 2 ;;
        --num-layers) NUM_LAYERS="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

echo "=== Neural SELPH LoRA Training ==="
echo "Model:      $MODEL"
echo "Data:       $DATA_DIR"
echo "Adapters:   $ADAPTER_DIR"
echo "Iters:      $ITERS"
echo "Batch size: $BATCH_SIZE"
echo "LR:         $LR"
echo "Layers:     $NUM_LAYERS"
echo ""

mkdir -p "$ADAPTER_DIR"

python -m mlx_lm lora \
    --model "$MODEL" \
    --data "$DATA_DIR" \
    --train \
    --fine-tune-type lora \
    --mask-prompt \
    --batch-size "$BATCH_SIZE" \
    --iters "$ITERS" \
    --learning-rate "$LR" \
    --num-layers "$NUM_LAYERS" \
    --adapter-path "$ADAPTER_DIR" \
    --steps-per-report 10 \
    --steps-per-eval 100 \
    --save-every 200 \
    --max-seq-length 512 \
    --test \
    "${EXTRA_ARGS[@]}"
