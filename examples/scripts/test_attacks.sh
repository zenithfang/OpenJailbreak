#!/bin/bash

# AutoJailbreak Attack Testing Script
# Uses the new 3-category attack system with secure environment variable handling

set -e  # Exit on any error

echo "============================================================"
echo "AutoJailbreak Attack Testing Script"
echo "============================================================"

# Configuration
MODEL_NAME="gpt-4o"
PROVIDER="openai"
EVAL_MODEL="gpt-4o"
EVAL_PROVIDER="azure"
DATASET="harmful"
SAMPLES=3
OUTPUT_DIR="results/test_$(date +%Y%m%d_%H%M%S)"

# Check for required environment variables
echo "Checking environment variables..."

if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ OPENAI_API_KEY not set. Please export your OpenAI API key:"
    echo "   export OPENAI_API_KEY='your-openai-key-here'"
    exit 1
fi

if [ -z "$AZURE_API_KEY" ] || [ -z "$AZURE_API_BASE" ]; then
    echo "❌ Azure OpenAI credentials not set. Please export:"
    echo "   export AZURE_API_KEY='your-azure-key-here'"
    echo "   export AZURE_API_BASE='https://your-resource.openai.azure.com'"
    exit 1
fi

echo "✅ Environment variables configured"
echo "✅ Attack model: $MODEL_NAME ($PROVIDER)"
echo "✅ Evaluation model: $EVAL_MODEL ($EVAL_PROVIDER)"
echo "✅ Output directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to run a single attack test
run_attack_test() {
    local attack_name="$1"
    local extra_args="$2"
    
    echo "Testing $attack_name attack..."
    python examples/universal_attack.py \
        --attack_name "$attack_name" \
        --model "$MODEL_NAME" \
        --provider "$PROVIDER" \
        --dataset "$DATASET" \
        --samples "$SAMPLES" \
        --eval_model "$EVAL_MODEL" \
        --eval_provider "$EVAL_PROVIDER" \
        --output_dir "$OUTPUT_DIR" \
        --verbose \
        $extra_args
    
    echo "✅ $attack_name test completed"
    echo ""
}

echo "==================================================================================="
echo "1. Testing Direct Attacks"
echo "==================================================================================="

# Test simple override attack
run_attack_test "simple_override"

# Test role assignment attack
run_attack_test "role_assignment"

# Test instruction injection
run_attack_test "instruction_injection"

echo "==================================================================================="
echo "2. Testing Obfuscation Attacks"
echo "==================================================================================="

# Test chain translation
run_attack_test "translate_chain" "--languages 'Spanish,German,English'"

# Test SATA academic
run_attack_test "sata_academic"

# Test many shot
run_attack_test "many_shot" "--num_examples 10"

echo "==================================================================================="
echo "3. Testing Iterative Attacks"
echo "==================================================================================="

# Test PAIR
run_attack_test "pair" "--max_rounds 3"

echo "==================================================================================="
echo "4. Testing Attack Listing"
echo "==================================================================================="

echo "Listing all available attacks..."
python examples/universal_attack.py --list_attacks

echo ""
echo "==================================================================================="
echo "🎉 ALL TESTS COMPLETED!"
echo "==================================================================================="

echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Summary of tests run:"
echo "  ✅ Direct attacks: simple_override, role_assignment, instruction_injection"
echo "  ✅ Obfuscation attacks: translate_chain, sata_academic, many_shot"
echo "  ✅ Iterative attacks: pair"
echo "  ✅ Attack listing functionality"
echo ""
echo "Check the results directory for detailed outputs and success rates."
echo ""
echo "Usage examples for future testing:"
echo "  # Run this script:"
echo "  ./examples/scripts/test_attacks.sh"
echo ""
echo "  # Test specific attack:"
echo "  python examples/universal_attack.py --attack_name simple_override --model gpt-4o"
echo ""
echo "  # Test specific attack with parameters:"
echo "  python examples/universal_attack.py --attack_name translate_chain --languages 'French,German,English' --model gpt-4o" 