#!/bin/bash

# AutoJailbreak Mousetrap Attack Testing Script

set -e  # Exit on any error

# Configuration
MODEL_NAME="gpt-4.1"
PROVIDER="wenwen"
EVAL_MODEL="gpt-4o"
EVAL_PROVIDER="wenwen"
DATASET="harmful"
SAMPLES=100
OUTPUT_DIR="results/test_mousetrap_$(date +%Y%m%d_%H%M%S)"

# Mousetrap specific parameters
MOUSETRAP_CHAOS_LENGTH=3


# Create output directory
mkdir -p "$OUTPUT_DIR"
echo "==================================================================================="
echo "Testing Mousetrap Attack"
echo "==================================================================================="

python examples/universal_attack.py \
    --attack_name "mousetrap" \
    --model "$MODEL_NAME" \
    --provider "$PROVIDER" \
    --dataset "$DATASET" \
    --samples "$SAMPLES" \
    --eval_model "$EVAL_MODEL" \
    --eval_provider "$EVAL_PROVIDER" \
    --output_dir "$OUTPUT_DIR" \
    --mousetrap_chaos_length "$MOUSETRAP_CHAOS_LENGTH" \
    --verbose

echo "✅ Mousetrap test completed"
echo "result saved to: $OUTPUT_DIR"
