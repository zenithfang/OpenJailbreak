#!/bin/bash

# AutoJailbreak QueryAttack Testing Script

set -e  # Exit on any error

# Source shared functions for result extraction
source "$(dirname "$0")/shared_functions.sh"

# Configuration
MODEL_NAME="gpt-oss-20b"
PROVIDER="infini"
EVAL_MODEL="gpt-4o"
EVAL_PROVIDER="wenwen"
DATASET="harmful"
SAMPLES=1
OUTPUT_DIR="results/test_query_attack_$(date +%Y%m%d_%H%M%S)"

# QueryAttack specific parameters
QUERY_ATTACK_TARGET_LANGUAGE="random"  # Choices: C++, C, C#, Python, Go, SQL, Java, JavaScript, URL, random
QUERY_ATTACK_TRANS_VERIFY="false"      # Use LLMs to verify each translation
QUERY_ATTACK_USE_ICL="true"            # Use In-Context Learning format


# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "==================================================================================="
echo "Testing QueryAttack"
echo "==================================================================================="

python examples/universal_attack.py \
    --attack_name "query_attack" \
    --model "$MODEL_NAME" \
    --provider "$PROVIDER" \
    --dataset "$DATASET" \
    --samples "$SAMPLES" \
    --eval_model "$EVAL_MODEL" \
    --eval_provider "$EVAL_PROVIDER" \
    --output_dir "$OUTPUT_DIR/python_test" \
    --query_attack_target_language "$QUERY_ATTACK_TARGET_LANGUAGE" \
    --query_attack_trans_verify "$QUERY_ATTACK_TRANS_VERIFY" \
    --query_attack_use_icl "$QUERY_ATTACK_USE_ICL" \
    --verbose

echo "✅ QueryAttack Python test completed"
echo "result saved to: $OUTPUT_DIR"

# Extract successful examples for analysis
echo ""
echo "🔍 Post-processing: Extracting successful jailbreak examples..."
if validate_extraction_script; then
    RESULT_FILE=$(find_result_file "$OUTPUT_DIR/python_test" "query_attack")
    if [ -n "$RESULT_FILE" ]; then
        extract_success_examples "$RESULT_FILE" "query_attack"
        print_extraction_summary "query_attack"
    else
        echo "⚠️  No result file found in $OUTPUT_DIR/python_test"
    fi
else
    echo "⚠️  Skipping extraction due to missing script"
fi
