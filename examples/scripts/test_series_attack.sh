#!/bin/bash

# Series attack test script

set -e  # Exit on error

echo "============================================================"
echo "AutoJailbreak Series Attack Test Script"
echo "============================================================"

# Configuration
MODEL_NAME="gpt-oss-20b"
PROVIDER="infini"
EVAL_MODEL="gpt-4o"
EVAL_PROVIDER="openai"
DATASET="harmful"
SAMPLES=50
OUTPUT_DIR="results/test_series_attack_$(date +%Y%m%d_%H%M%S)"

echo "Configuration:"
echo "  📊 Target dataset: $DATASET"
echo "  🤖 Target model: $MODEL_NAME ($PROVIDER)"
echo "  🔬 Evaluation model: $EVAL_MODEL ($EVAL_PROVIDER)"
echo "  📁 Number of samples: $SAMPLES"
echo "  📂 Output directory: $OUTPUT_DIR"
echo ""

# Check required environment variables
echo "Checking environment variables..."

if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ OPENAI_API_KEY is not set. Please export your OpenAI API key for evaluation:"
    echo "   export OPENAI_API_KEY='your-openai-key-here'"
    exit 1
fi

if [ -z "$WENWEN_API_KEY" ]; then
    echo "❌ WENWEN_API_KEY is not set. Please export your Wenwen API key:"
    echo "   export WENWEN_API_KEY='your-wenwen-key-here'"
    exit 1
fi

echo "✅ Environment variables configured"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "==================================================================================="
echo "Test: Series Attack"
echo "==================================================================================="

echo "Running simple series attack..."
python examples/universal_attack_series.py \
    --config_file "assets/series_configs/series_simple.json" \
    --model "$MODEL_NAME" \
    --provider "$PROVIDER" \
    --dataset "$DATASET" \
    --samples "$SAMPLES" \
    --eval_model "$EVAL_MODEL" \
    --eval_provider "$EVAL_PROVIDER" \
    --output_dir "$OUTPUT_DIR/series" \
    --verbose

echo "✅ Series attack test completed"
echo ""

# echo "==================================================================================="
# echo "Test 2: Advanced series attack (multi-shot + deep nesting + translation)"
# echo "==================================================================================="

# echo "Running advanced series attack..."
# python examples/universal_attack_series.py \
#     --config_file "examples/configs/series_advanced.json" \
#     --model "$MODEL_NAME" \
#     --provider "$PROVIDER" \
#     --dataset "$DATASET" \
#     --samples 2 \
#     --eval_model "$EVAL_MODEL" \
#     --eval_provider "$EVAL_PROVIDER" \
#     --output_dir "$OUTPUT_DIR/advanced_series" \
#     --verbose

# echo "✅ Advanced series attack test completed"
# echo ""

# echo "==================================================================================="
# echo "Test 3: Psychology-oriented series attack"
# echo "==================================================================================="

# echo "Running psychology-oriented series attack..."
# python examples/universal_attack_series.py \
#     --config_file "examples/configs/series_psychological.json" \
#     --model "$MODEL_NAME" \
#     --provider "$PROVIDER" \
#     --dataset "$DATASET" \
#     --samples 2 \
#     --eval_model "$EVAL_MODEL" \
#     --eval_provider "$EVAL_PROVIDER" \
#     --output_dir "$OUTPUT_DIR/psychological_series" \
#     --verbose

# echo "✅ Psychology-oriented series attack test completed"
# echo ""

# echo "==================================================================================="
# echo "Test 4: Command-line attack chain test"
# echo "==================================================================================="

# echo "Testing a simple command-line attack chain..."
# python examples/universal_attack_series.py \
#     --attack_chain "simple_override,deepinception" \
#     --model "$MODEL_NAME" \
#     --provider "$PROVIDER" \
#     --dataset "$DATASET" \
#     --samples 2 \
#     --eval_model "$EVAL_MODEL" \
#     --eval_provider "$EVAL_PROVIDER" \
#     --output_dir "$OUTPUT_DIR/cmdline_series" \
#     --verbose

# echo "✅ Command-line attack chain test completed"
# echo ""

# echo "==================================================================================="
# echo "Test 5: Dry-run mode test"
# echo "==================================================================================="

# echo "Running dry-run mode test..."
# python examples/universal_attack_series.py \
#     --config_file "examples/configs/series_simple.json" \
#     --model "$MODEL_NAME" \
#     --provider "$PROVIDER" \
#     --dataset "$DATASET" \
#     --samples 1 \
#     --dry_run \
#     --verbose

# echo "✅ Dry-run mode test completed"
# echo ""

# echo "==================================================================================="
# echo "🎉 All tests completed!"
# echo "==================================================================================="

# echo "Results saved to: $OUTPUT_DIR"
# echo ""
# echo "Test summary:"
# echo "  ✅ Simple series attack: simple_override -> translate_attack"
# echo "  ✅ Advanced series attack: many_shot -> deepinception -> translate_attack"
# echo "  ✅ Psychology-based attack: cognitive_hacking -> past_tense -> instruction_repetition"
# echo "  ✅ Command-line attack chain: simple_override -> deepinception"
# echo "  ✅ Dry-run mode: no actual attack executed"
# echo ""
# echo "Configuration:"
# echo "  🤖 Target model: $PROVIDER/$MODEL_NAME"
# echo "  📊 Evaluator: $EVAL_PROVIDER/$EVAL_MODEL"
# echo "  📊 Dataset: $DATASET"
# echo ""
# echo "Usage examples:"
# echo "  # Run this script:"
# echo "  ./examples/scripts/test_series_attack.sh"
# echo ""
# echo "  # Use custom sample count:"
# echo "  ./examples/scripts/test_series_attack.sh 5"
# echo ""
# echo "  # Use configuration file:"
# echo "  python examples/universal_attack_series.py --config_file examples/configs/series_simple.json"
# echo ""
# echo "  # Use command-line attack chain:"
# echo "  python examples/universal_attack_series.py --attack_chain 'simple_override,translate_attack' --samples 3"
# echo ""
# echo "Expected effectiveness of series attacks:"
# echo "  📈 Simple series: Moderate effect - override + language bypass"
# echo "  📈 Advanced series: High effect - context + nesting + language"
# echo "  📈 Psychology series: Medium-high effect - leveraging cognitive biases"
# echo "  📈 Command-line series: Configurable effect"
