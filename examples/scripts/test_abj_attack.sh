#!/bin/bash

# AutoJailbreak ABJ (Analyzing-based Jailbreak) Attack Testing Script
set -e

# Configuration
MODEL_NAME="gpt-4o-mini"
PROVIDER="wenwen"
EVAL_MODEL="gpt-4o"
EVAL_PROVIDER="wenwen"
DATASET="harmful"
SAMPLES=100
OUTPUT_DIR="results/test_abj_$(date +%Y%m%d_%H%M%S)"

# ABJ specific parameters
ABJ_ATTACKER_MODEL="qwen2.5-32b"           # Assistant model for data transformation
ABJ_ATTACKER_BASE_URL="http://10.210.22.10:30255/v1"
ABJ_ATTACKER_API_KEY=""
ABJ_ATTACKER_PROVIDER="openai"             # Provider for attacker model
ABJ_JUDGE_MODEL="gpt-4o"                   # Judge model for response evaluation
ABJ_JUDGE_PROVIDER="wenwen"                # Provider for judge model
ABJ_MAX_ATTACK_ROUNDS="3"                  # Number of attack rounds
ABJ_MAX_ADJUSTMENT_ROUNDS="5"              # Maximum toxicity adjustment rounds


# Create output directory
mkdir -p "$OUTPUT_DIR"
echo "==================================================================================="
echo "Testing Analyzing-based Jailbreak Attack (ABJ)"
echo "==================================================================================="

python examples/universal_attack.py \
    --attack_name "abj" \
    --model "$MODEL_NAME" \
    --provider "$PROVIDER" \
    --dataset "$DATASET" \
    --samples "$SAMPLES" \
    --eval_model "$EVAL_MODEL" \
    --eval_provider "$EVAL_PROVIDER" \
    --output_dir "$OUTPUT_DIR" \
    --abj_attacker_model "$ABJ_ATTACKER_MODEL" \
    --abj_attacker_base_url "$ABJ_ATTACKER_BASE_URL" \
    --abj_attacker_api_key "$ABJ_ATTACKER_API_KEY" \
    --abj_attacker_provider "$ABJ_ATTACKER_PROVIDER" \
    --abj_judge_model "$ABJ_JUDGE_MODEL" \
    --abj_judge_provider "$ABJ_JUDGE_PROVIDER" \
    --abj_max_attack_rounds "$ABJ_MAX_ATTACK_ROUNDS" \
    --abj_max_adjustment_rounds "$ABJ_MAX_ADJUSTMENT_ROUNDS" \
    --verbose

echo "✅ ABJ attack test completed successfully."
echo "Results saved to: $OUTPUT_DIR"
