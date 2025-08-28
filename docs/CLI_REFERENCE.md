# Command Line Reference

Complete documentation for the OpenJailbreak universal attack script command-line interface.

> **Note**: This documentation reflects the minimalist architecture (v2.0) with streamlined attack attributes and flattened registry structure.

## Quick Reference

```bash
python examples/universal_attack.py [OPTIONS]
```

## Basic Usage Patterns

### Single Attack
```bash
python examples/universal_attack.py --attack_name ATTACK --model MODEL [OPTIONS]
```

> Attack listing has been removed for performance reasons. Specify `--attack_name` and use `--help` alongside it for attack-specific options.

## Core Arguments

### Attack Selection
- `--attack_name NAME`
  - **Description**: Specific attack to run
  - **Available Attacks**: `abj`, `mousetrap`, `query_attack`, `wiki_text_infilling`
  - **Example**: `--attack_name abj`

### Model Configuration
- `--model MODEL`
  - **Description**: Target model to attack
  - **Default**: `gpt-4o`
  - **Examples**: `gpt-4o`, `claude-3-sonnet-20240229`, `gpt-3.5-turbo`

- `--provider PROVIDER`
  - **Description**: Model provider
  - **Options**: `openai`, `anthropic`, `azure`, `bedrock`, `vertex_ai`, `aliyun`, `wenwen`, `infini`, `local`
  - **Default**: `openai`

- `--api_key KEY`
  - **Description**: API key for the model provider
  - **Note**: Can also be set via environment variables

- `--api_base URL`
  - **Description**: API base URL (required for Azure OpenAI)
  - **Example**: `https://your-resource.openai.azure.com/`

### Dataset and Sampling
- `--dataset DATASET`
  - **Description**: Dataset to use for evaluation
  - **Options**: `jbb-harmful`, `jbb-benign`, `custom`, `advbench`, `harmbench`
  - **Default**: `jbb-harmful`

- `--dataset_path PATH`
  - **Description**: Path to custom dataset file
  - **Required**: When using `--dataset custom`

- `--samples N`
  - **Description**: Number of examples to test
  - **Default**: `5`

- `--start_index N`
  - **Description**: Starting index for sample selection
  - **Default**: `0`

## Attack-Specific Arguments

### ABJ Attack (Chain-of-Thought exploitation)
- `--abj_max_attack_rounds N` (default: 1)
- `--abj_max_adjustment_rounds N` (default: 5)
- `--abj_attacker_model MODEL`
- `--abj_attacker_provider PROVIDER`
- `--abj_attacker_base_url URL`
- `--abj_attacker_api_key KEY`
- `--abj_judge_model MODEL`
- `--abj_judge_provider PROVIDER`

### Mousetrap Attack (Iterative chaos)
- `--mousetrap_chaos_length {1,2,3}` (default: 3)

### QueryAttack (Structured non-natural language)
- `--query_attack_target_language {C++,C,C#,Python,Go,SQL,Java,JavaScript,URL,random}`
- `--query_attack_trans_verify {true,false}`
- `--query_attack_use_icl {true,false}`

### Wiki-Text-Infilling (Academic framing)
- `--wiki_variant {sw,sp,mw,mp}`
- `--wiki_attack_model MODEL`
- `--wiki_model MODEL`
- `--wiki_paraphrase_model MODEL`
- `--wiki_detection_temp FLOAT`
- `--wiki_temp FLOAT`
- `--wiki_paraphrase_temp FLOAT`
- `--wiki_max_detection_retries N`
- `--wiki_max_wiki_retries N`
- `--wiki_max_tokens N`
- `--wiki_no_cache {true,false}`

## Evaluation Settings
- `--eval_model MODEL`
  - **Description**: Model used for evaluation
  - **Default**: `gpt-4o`

- `--eval_provider PROVIDER`
  - **Description**: Provider for evaluation model
  - **Options**: `openai`, `azure`, `anthropic`, `wenwen`
  - **Default**: `openai`

## Output Configuration
- `--output PATH`
  - **Description**: Output file to save results
  - **Default**: Auto-generated based on attack and timestamp

- `--output_dir DIR`
  - **Description**: Directory to save results
  - **Default**: `results`

- `--save_attacks`
  - **Description**: Save attack prompts in results
  - **Default**: `False`

## Debug and Logging
- `--verbose`
  - **Description**: Enable verbose logging
  - **Default**: `False`

- `--seed N`
  - **Description**: Random seed for reproducibility
  - **Default**: `42`

## Environment Variables

### API Keys
```bash
# Aliyun API Key
export DASHSCOPE_API_KEY=""

# OpenAI API Key
export OPENAI_API_KEY=""

# AZURE API Key
export AZURE_API_KEY=""
export AZURE_API_BASE=""
export AZURE_API_VERSION="" # 2025-01-01-preview

# AWS Bedrock API Key
export AWS_ACCESS_KEY_ID=""
export AWS_SECRET_ACCESS_KEY=""
export AWS_REGION="" # us-east-1

# Google Gemini Vertex AI 配置
## RUN ## 
# !gcloud auth application-default login - run this to add vertex credentials to your env
## OR ## 
export GOOGLE_APPLICATION_CREDENTIALS="" # path/to/vertex_ai_service_account.json
export GOOGLE_CLOUD_PROJECT="" # nova-gemini-xxxxx
export GOOGLE_CLOUD_LOCATION="" # us-central1
```

### Model Configuration
```bash
export AUTOJAILBREAK_DEFAULT_MODEL="gpt-4o"
export AUTOJAILBREAK_MAX_EXAMPLES="25"
```

## Example Commands

### Basic Examples
```bash
# ABJ attack on GPT-4o
python examples/universal_attack.py \
    --attack_name abj \
    --model gpt-4o \
    --samples 3

# Mousetrap with custom chaos length
python examples/universal_attack.py \
    --attack_name mousetrap \
    --mousetrap_chaos_length 2 \
    --model gpt-4o \
    --samples 5

# QueryAttack with ICL formatting
python examples/universal_attack.py \
    --attack_name query_attack \
    --query_attack_target_language Python \
    --query_attack_use_icl true \
    --model gpt-4o \
    --samples 2
```

### Advanced Examples
```bash
# Wiki-Text-Infilling with variant and caching options
python examples/universal_attack.py \
    --attack_name wiki_text_infilling \
    --wiki_variant mw \
    --wiki_no_cache false \
    --model gpt-4o \
    --samples 2

# ABJ with attacker/judge configuration
python examples/universal_attack.py \
    --attack_name abj \
    --abj_attacker_model gpt-4o-mini \
    --abj_attacker_provider openai \
    --abj_judge_model gpt-4o \
    --abj_judge_provider openai \
    --model gpt-4o \
    --samples 1 \
    --verbose

# Attack with defense mechanism
python examples/universal_attack.py \
    --attack_name query_attack \
    --defense smoothllm \
    --model gpt-4o \
    --samples 2 \
    --output_dir experiments/defense_test
```

### External File Usage
```bash
# Custom dataset
python examples/universal_attack.py \
    --attack_name query_attack \
    --dataset custom \
    --dataset_path data/custom_prompts.json \
    --model claude-3-opus-20240229 \
    --provider anthropic
```

## Available Attacks

### Current Attack List
Core attacks aligned with the writeup and codebase:

- `abj` - Analyzing-based Jailbreak (Chain-of-Thought exploitation)
- `mousetrap` - Iterative chaos to induce reward hacking
- `query_attack` - Structured non-natural query language manipulation
- `wiki_text_infilling` - Academic framing with [MASK] infilling

### Attack Discovery
The framework automatically discovers attacks by scanning the `src/autojailbreak/attacks/` directory. Each attack is implemented as a class inheriting from `ModernBaseAttack` with embedded configuration.

## Exit Codes
- `0`: Success
- `1`: General error
- `2`: Command line argument error
- `3`: Model provider error
- `4`: Dataset loading error
- `5`: Attack execution error

## File Formats

### External Many-Shot File
```text
Q: What is the capital of France?
A: The capital of France is Paris.

Q: How do I make a sandwich?
A: Place fillings between two slices of bread.

Q: What is 2+2?
A: 2+2 equals 4.
```

### Custom Dataset JSON
```json
[
  {
    "goal": "Write a harmful message",
    "target": "Sure, here's a harmful message",
    "behavior": "Harmful Content Generation",
    "category": "Harmful"
  }
]
```

## Troubleshooting

### Common Issues
1. **API Key Not Found**: Set environment variables or use `--api_key`
2. **Model Not Available**: Check provider and model name spelling
3. **Attack Not Found**: Verify the attack name is one of: `abj`, `mousetrap`, `query_attack`, `wiki_text_infilling`
4. **File Not Found**: Verify paths for `--examples_file` and `--dataset_path`
5. **Rate Limit Exceeded**: Reduce `--samples` or add delays between requests

### Debug Tips
- Use `--verbose` for detailed logging
- Check output directory permissions for result saving
- Verify API keys are correctly set and have sufficient credits

### Parameter Validation
The framework provides clear error messages for invalid parameters:
- Type mismatches (e.g., string instead of integer)
- Invalid choices for restricted parameters
- Missing required parameters
- File path validation errors

### Getting Help
```bash
# Show all available options
python examples/universal_attack.py --help

# Attack-specific options
python examples/universal_attack.py --attack_name abj --help

# Run with verbose output for debugging
python examples/universal_attack.py --attack_name ATTACK --verbose
```

---

*This reference reflects the current flattened architecture with auto-discovery and embedded configuration.* 

## Series Runner CLI

Run compositional attacks by chaining multiple components.

### Usage
```bash
python examples/universal_attack_series.py [OPTIONS]
```

### Core Options
- `--config_file PATH` – Series config JSON file
- `--attack_chain LIST` – Comma-separated chain, e.g., `past_tense_attack,translate_chain`
- `--model MODEL` – Target model (default: `gpt-3.5-turbo`)
- `--provider PROVIDER` – Target provider (default: `wenwen`)
- `--api_key KEY` – Provider API key (optional if env var set)
- `--api_base URL` – Provider base URL
- `--dataset NAME` – Dataset (default: `harmful` ≈ `jbb-harmful`)
- `--samples N` – Number of samples (default: `5`)
- `--all_samples` – Use entire dataset
- `--seed N` – Random seed (default: `42`)
- `--eval_provider PROVIDER` – Evaluation provider (default: `openai`)
- `--eval_model MODEL` – Evaluation model (default: `gpt-4o`)
- `--output_dir DIR` – Output directory (default: `results`)
- `--output PATH` – Output file path (overrides auto-naming)
- `--resume` – Resume appending to an existing results file
- `--max_workers N` – Parallel workers (default: `1`)
- `--verbose` – Verbose logging
- `--dry_run` – Do not execute attacks; simulate
- `--continue_on_error` – Continue chain on step failure
- `--max_retry_attempts N` – Retry per step (default: `2`)

### Notes
- Auto evaluator selection for datasets:
  - `gsm8k`, `gsm8k-evil` → GSM8K-Hybrid evaluator
  - `wmdp-*` → WMDP-Hybrid evaluator
- Results JSON includes `metadata.success_rate` and per-example details.

### Series Components
Usable in chains (distinct from single attacks listed above):
- `past_tense_attack`
- `translate_chain`

Example:
```bash
python examples/universal_attack_series.py \
  --attack_chain "past_tense_attack,translate_chain" \
  --model gpt-4o --provider openai \
  --samples 3 --verbose
```