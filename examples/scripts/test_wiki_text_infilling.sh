#!/bin/bash

# AutoJailbreak Wiki-Text-Infilling Attack Testing Script
# SATA Wiki-Text-Infilling Attack - Testing all four variants: SW, SP, MW, MP

set -e  # Exit on any error

echo "============================================================"
echo "AutoJailbreak Wiki-Text-Infilling Attack Testing Script"
echo "============================================================"

# Configuration
MODEL_NAME="gpt-4o"
PROVIDER="openai"
EVAL_MODEL="gpt-4o"
EVAL_PROVIDER="openai"
DATASET="harmful"
SAMPLES=3
OUTPUT_DIR="results/test_wiki_text_infilling_$(date +%Y%m%d_%H%M%S)"

# Wiki-Text-Infilling specific parameters with defaults
WIKI_VARIANT="sw"                         # Wiki variant: sw | sp | mw | mp
WIKI_ATTACK_MODEL="gpt-4o"               # Model for keyword detection/masking
WIKI_MODEL="gpt-4o"                      # Model for Wikipedia entry generation
WIKI_PARAPHRASE_MODEL="gpt-3.5-turbo"   # Model for paraphrasing instructions
WIKI_DETECTION_TEMP="0.9"               # Temperature for keyword detection
WIKI_TEMP="0.9"                         # Temperature for Wikipedia generation
WIKI_PARAPHRASE_TEMP="0.9"              # Temperature for paraphrasing
MAX_DETECTION_RETRIES="5"               # Max retries for keyword detection
MAX_WIKI_RETRIES="8"                    # Max retries for wiki generation
MAX_TOKENS="10000"                      # Maximum tokens for generated content
CACHE_ENABLED="true"                     # Enable caching of generated content

# Check for required environment variables
echo "Checking environment variables..."

if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ OPENAI_API_KEY not set. Please export your OpenAI API key:"
    echo "   export OPENAI_API_KEY='your-openai-key-here'"
    exit 1
fi

echo "✅ Environment variables configured"
echo "✅ Attack model: $MODEL_NAME ($PROVIDER)"
echo "✅ Evaluation model: $EVAL_MODEL ($EVAL_PROVIDER)"
echo "✅ Wiki attack model: $WIKI_ATTACK_MODEL"
echo "✅ Wiki generation model: $WIKI_MODEL"
echo "✅ Wiki paraphrase model: $WIKI_PARAPHRASE_MODEL"
echo "✅ Output directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "==================================================================================="
echo "Wiki-Text-Infilling Algorithm Core Components:"
echo "==================================================================================="
echo "🎯 SW (Single Word):      Masks one harmful word and creates Wikipedia entries"
echo "🎯 SP (Single Phrase):    Masks one harmful phrase and creates Wikipedia entries"
echo "🎯 MW (Multiple Words):   Masks multiple harmful words and creates Wikipedia entries"
echo "🎯 MP (Multiple Phrases): Masks multiple harmful phrases and creates Wikipedia entries"
echo "📝 LLM-based Masking:     Uses LLM to identify and mask harmful keywords/phrases"
echo "📖 Wiki Generation:       Creates contextual Wikipedia entries for masked concepts"
echo "🔄 Text Infilling:        Structures content into MLM-style completion format"
echo "💾 Intelligent Caching:   Caches generated content for efficiency"
echo ""

echo "==================================================================================="
echo "Testing Wiki-Text-Infilling Attack - Variant 1: Single Word (SW)"
echo "==================================================================================="

echo "Wiki-Text-Infilling SW Parameters Configuration:"
echo "  🎯 Variant:               $WIKI_VARIANT"
echo "  🔍 Attack Model:          $WIKI_ATTACK_MODEL"
echo "  📖 Wiki Model:            $WIKI_MODEL"
echo "  📝 Paraphrase Model:      $WIKI_PARAPHRASE_MODEL"
echo "  🌡️  Detection Temp:        $WIKI_DETECTION_TEMP"
echo "  🌡️  Wiki Temp:             $WIKI_TEMP"
echo "  🌡️  Paraphrase Temp:       $WIKI_PARAPHRASE_TEMP"
echo "  🔄 Max Retries:           Detection: $MAX_DETECTION_RETRIES, Wiki: $MAX_WIKI_RETRIES"
echo "  📏 Max Tokens:            $MAX_TOKENS"
echo "  💾 Cache Enabled:         $CACHE_ENABLED"
echo ""

echo "Testing Wiki-Text-Infilling with Single Word variant..."
python examples/universal_attack.py \
    --attack_name "wiki_text_infilling" \
    --model "$MODEL_NAME" \
    --provider "$PROVIDER" \
    --dataset "$DATASET" \
    --samples "$SAMPLES" \
    --eval_model "$EVAL_MODEL" \
    --eval_provider "$EVAL_PROVIDER" \
    --output_dir "$OUTPUT_DIR/sw_test" \
    --wiki_variant "sw" \
    --wiki_attack_model "$WIKI_ATTACK_MODEL" \
    --wiki_model "$WIKI_MODEL" \
    --wiki_paraphrase_model "$WIKI_PARAPHRASE_MODEL" \
    --wiki_detection_temp "$WIKI_DETECTION_TEMP" \
    --wiki_temp "$WIKI_TEMP" \
    --wiki_paraphrase_temp "$WIKI_PARAPHRASE_TEMP" \
    --wiki_max_detection_retries "$MAX_DETECTION_RETRIES" \
    --wiki_max_wiki_retries "$MAX_WIKI_RETRIES" \
    --wiki_max_tokens "$MAX_TOKENS" \
    --wiki_no_cache false \
    --verbose

echo "✅ Wiki-Text-Infilling Single Word test completed"
echo ""

echo "==================================================================================="
echo "Testing Wiki-Text-Infilling Attack - Variant 2: Single Phrase (SP)"
echo "==================================================================================="

echo "Testing Wiki-Text-Infilling with Single Phrase variant..."
python examples/universal_attack.py \
    --attack_name "wiki_text_infilling" \
    --model "$MODEL_NAME" \
    --provider "$PROVIDER" \
    --dataset "$DATASET" \
    --samples "$SAMPLES" \
    --eval_model "$EVAL_MODEL" \
    --eval_provider "$EVAL_PROVIDER" \
    --output_dir "$OUTPUT_DIR/sp_test" \
    --wiki_variant "sp" \
    --wiki_attack_model "$WIKI_ATTACK_MODEL" \
    --wiki_model "$WIKI_MODEL" \
    --wiki_paraphrase_model "$WIKI_PARAPHRASE_MODEL" \
    --wiki_detection_temp "$WIKI_DETECTION_TEMP" \
    --wiki_temp "$WIKI_TEMP" \
    --wiki_paraphrase_temp "$WIKI_PARAPHRASE_TEMP" \
    --wiki_max_detection_retries "$MAX_DETECTION_RETRIES" \
    --wiki_max_wiki_retries "$MAX_WIKI_RETRIES" \
    --wiki_max_tokens "$MAX_TOKENS" \
    --wiki_no_cache false \
    --verbose

echo "✅ Wiki-Text-Infilling Single Phrase test completed"
echo ""

echo "==================================================================================="
echo "Testing Wiki-Text-Infilling Attack - Variant 3: Multiple Words (MW)"
echo "==================================================================================="

echo "Testing Wiki-Text-Infilling with Multiple Words variant..."
python examples/universal_attack.py \
    --attack_name "wiki_text_infilling" \
    --model "$MODEL_NAME" \
    --provider "$PROVIDER" \
    --dataset "$DATASET" \
    --samples "$SAMPLES" \
    --eval_model "$EVAL_MODEL" \
    --eval_provider "$EVAL_PROVIDER" \
    --output_dir "$OUTPUT_DIR/mw_test" \
    --wiki_variant "mw" \
    --wiki_attack_model "$WIKI_ATTACK_MODEL" \
    --wiki_model "$WIKI_MODEL" \
    --wiki_paraphrase_model "$WIKI_PARAPHRASE_MODEL" \
    --wiki_detection_temp "$WIKI_DETECTION_TEMP" \
    --wiki_temp "$WIKI_TEMP" \
    --wiki_paraphrase_temp "$WIKI_PARAPHRASE_TEMP" \
    --wiki_max_detection_retries "$MAX_DETECTION_RETRIES" \
    --wiki_max_wiki_retries "$MAX_WIKI_RETRIES" \
    --wiki_max_tokens "$MAX_TOKENS" \
    --wiki_no_cache false \
    --verbose

echo "✅ Wiki-Text-Infilling Multiple Words test completed"
echo ""

echo "==================================================================================="
echo "Testing Wiki-Text-Infilling Attack - Variant 4: Multiple Phrases (MP)"
echo "==================================================================================="

echo "Testing Wiki-Text-Infilling with Multiple Phrases variant..."
python examples/universal_attack.py \
    --attack_name "wiki_text_infilling" \
    --model "$MODEL_NAME" \
    --provider "$PROVIDER" \
    --dataset "$DATASET" \
    --samples "$SAMPLES" \
    --eval_model "$EVAL_MODEL" \
    --eval_provider "$EVAL_PROVIDER" \
    --output_dir "$OUTPUT_DIR/mp_test" \
    --wiki_variant "mp" \
    --wiki_attack_model "$WIKI_ATTACK_MODEL" \
    --wiki_model "$WIKI_MODEL" \
    --wiki_paraphrase_model "$WIKI_PARAPHRASE_MODEL" \
    --wiki_detection_temp "$WIKI_DETECTION_TEMP" \
    --wiki_temp "$WIKI_TEMP" \
    --wiki_paraphrase_temp "$WIKI_PARAPHRASE_TEMP" \
    --wiki_max_detection_retries "$MAX_DETECTION_RETRIES" \
    --wiki_max_wiki_retries "$MAX_WIKI_RETRIES" \
    --wiki_max_tokens "$MAX_TOKENS" \
    --wiki_no_cache false \
    --verbose

echo "✅ Wiki-Text-Infilling Multiple Phrases test completed"
echo ""

echo "==================================================================================="
echo "Testing Wiki-Text-Infilling Attack - High-Performance Configuration"
echo "==================================================================================="

echo "Testing Wiki-Text-Infilling with high-performance configuration (higher retry limits)..."
python examples/universal_attack.py \
    --attack_name "wiki_text_infilling" \
    --model "$MODEL_NAME" \
    --provider "$PROVIDER" \
    --dataset "$DATASET" \
    --samples "$SAMPLES" \
    --eval_model "$EVAL_MODEL" \
    --eval_provider "$EVAL_PROVIDER" \
    --output_dir "$OUTPUT_DIR/high_performance_test" \
    --wiki_variant "mw" \
    --wiki_attack_model "$WIKI_ATTACK_MODEL" \
    --wiki_model "$WIKI_MODEL" \
    --wiki_paraphrase_model "$WIKI_PARAPHRASE_MODEL" \
    --wiki_detection_temp "0.95" \
    --wiki_temp "0.95" \
    --wiki_paraphrase_temp "0.8" \
    --wiki_max_detection_retries 10 \
    --wiki_max_wiki_retries 15 \
    --wiki_max_tokens 15000 \
    --wiki_no_cache false \
    --verbose

echo "✅ Wiki-Text-Infilling High-performance test completed"
echo ""

echo "==================================================================================="
echo "Testing Wiki-Text-Infilling Attack - No-Cache Configuration"
echo "==================================================================================="

echo "Testing Wiki-Text-Infilling with caching disabled (fresh generation each time)..."
python examples/universal_attack.py \
    --attack_name "wiki_text_infilling" \
    --model "$MODEL_NAME" \
    --provider "$PROVIDER" \
    --dataset "$DATASET" \
    --samples 2 \
    --eval_model "$EVAL_MODEL" \
    --eval_provider "$EVAL_PROVIDER" \
    --output_dir "$OUTPUT_DIR/no_cache_test" \
    --wiki_variant "sw" \
    --wiki_attack_model "$WIKI_ATTACK_MODEL" \
    --wiki_model "$WIKI_MODEL" \
    --wiki_paraphrase_model "$WIKI_PARAPHRASE_MODEL" \
    --wiki_detection_temp "$WIKI_DETECTION_TEMP" \
    --wiki_temp "$WIKI_TEMP" \
    --wiki_paraphrase_temp "$WIKI_PARAPHRASE_TEMP" \
    --wiki_max_detection_retries "$MAX_DETECTION_RETRIES" \
    --wiki_max_wiki_retries "$MAX_WIKI_RETRIES" \
    --wiki_max_tokens "$MAX_TOKENS" \
    --wiki_no_cache true \
    --verbose

echo "✅ Wiki-Text-Infilling No-cache test completed"
echo ""

echo "==================================================================================="
echo "Testing Wiki-Text-Infilling Attack - Mixed Models Configuration"
echo "==================================================================================="

echo "Testing Wiki-Text-Infilling with mixed model configuration..."
python examples/universal_attack.py \
    --attack_name "wiki_text_infilling" \
    --model "$MODEL_NAME" \
    --provider "$PROVIDER" \
    --dataset "$DATASET" \
    --samples "$SAMPLES" \
    --eval_model "$EVAL_MODEL" \
    --eval_provider "$EVAL_PROVIDER" \
    --output_dir "$OUTPUT_DIR/mixed_models_test" \
    --wiki_variant "sp" \
    --wiki_attack_model "gpt-3.5-turbo" \
    --wiki_model "gpt-4o" \
    --wiki_paraphrase_model "gpt-3.5-turbo" \
    --wiki_detection_temp "0.8" \
    --wiki_temp "1.0" \
    --wiki_paraphrase_temp "0.7" \
    --wiki_max_detection_retries "$MAX_DETECTION_RETRIES" \
    --wiki_max_wiki_retries "$MAX_WIKI_RETRIES" \
    --wiki_max_tokens 8000 \
    --wiki_no_cache false \
    --verbose

echo "✅ Wiki-Text-Infilling Mixed models test completed"
echo ""

echo "==================================================================================="
echo "🎉 ALL WIKI-TEXT-INFILLING TESTS COMPLETED!"
echo "==================================================================================="

echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Test summary:"
echo "  ✅ Single Word (SW) variant test"
echo "  ✅ Single Phrase (SP) variant test"
echo "  ✅ Multiple Words (MW) variant test"
echo "  ✅ Multiple Phrases (MP) variant test"
echo "  ✅ High-performance configuration test"
echo "  ✅ No-cache configuration test"
echo "  ✅ Mixed models configuration test"
echo ""
echo "Wiki-Text-Infilling Algorithm Summary:"
echo "  🎯 Four Core Variants:"
echo "    - SW: Masks single harmful words and generates Wikipedia entries"
echo "    - SP: Masks single harmful phrases and generates Wikipedia entries"
echo "    - MW: Masks multiple harmful words and generates Wikipedia entries"
echo "    - MP: Masks multiple harmful phrases and generates Wikipedia entries"
echo ""
echo "  📝 Four-Step Process:"
echo "    1. LLM-based masking of harmful keywords/phrases"
echo "    2. Contextual Wikipedia entry generation for masked concepts"
echo "    3. Paraphrasing to wiki-style format"
echo "    4. Text infilling prompt construction with MLM-style completion"
echo ""
echo "  🔧 Advanced Features:"
echo "    - Intelligent caching system for generated content"
echo "    - Configurable retry limits for robustness"
echo "    - Multiple LLM specialization (detection, wiki generation, paraphrasing)"
echo "    - Token length management and optimization"
echo ""
echo "Usage examples for future testing:"
echo ""
echo "  # Standard Wiki-Text-Infilling (Single Word variant):"
echo "  python examples/universal_attack.py --attack_name wiki_text_infilling \\"
echo "    --wiki_variant sw --wiki_max_tokens 10000 --model gpt-4o"
echo ""
echo "  # Multiple Words variant with high-performance settings:"
echo "  python examples/universal_attack.py --attack_name wiki_text_infilling \\"
echo "    --wiki_variant mw --wiki_max_wiki_retries 15 \\"
echo "    --wiki_temp 0.95 --model gpt-4o"
echo ""
echo "  # Mixed models configuration for cost optimization:"
echo "  python examples/universal_attack.py --attack_name wiki_text_infilling \\"
echo "    --wiki_variant sp --wiki_attack_model gpt-3.5-turbo \\"
echo "    --wiki_model gpt-4o --model gpt-4o"
echo ""
echo "  # No-cache testing for fresh generation:"
echo "  python examples/universal_attack.py --attack_name wiki_text_infilling \\"
echo "    --wiki_variant sw --wiki_no_cache true --model gpt-4o"
echo ""
echo "Complete Wiki-Text-Infilling parameter reference:"
echo "  --wiki_variant                [sw|sp|mw|mp] (default: sw)"
echo "  --wiki_attack_model           Model for keyword detection (default: gpt-4o)"
echo "  --wiki_model                  Model for Wikipedia generation (default: gpt-4o)"
echo "  --wiki_paraphrase_model       Model for paraphrasing (default: gpt-3.5-turbo)"
echo "  --wiki_detection_temp         Detection temperature (default: 0.9)"
echo "  --wiki_temp                   Wiki generation temperature (default: 0.9)"
echo "  --wiki_paraphrase_temp        Paraphrasing temperature (default: 0.9)"
echo "  --wiki_max_detection_retries  Max detection retries (default: 5)"
echo "  --wiki_max_wiki_retries       Max wiki generation retries (default: 8)"
echo "  --wiki_max_tokens             Max tokens for content (default: 10000)"
echo "  --wiki_no_cache               Disable caching [true|false] (default: false)"