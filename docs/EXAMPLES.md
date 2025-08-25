# Examples

Comprehensive collection of OpenJailbreak usage examples for research and development.

## Basic Attack Examples

### ABJ Attack (Chain-of-Thought exploitation)
```bash
python examples/universal_attack.py \
    --attack_name abj \
    --model gpt-4o \
    --samples 3
```

### Mousetrap Attack (Iterative chaos)
```bash
python examples/universal_attack.py \
    --attack_name mousetrap \
    --mousetrap_chaos_length 2 \
    --model gpt-4o \
    --samples 3
```

### QueryAttack (Structured non-natural language)
```bash
python examples/universal_attack.py \
    --attack_name query_attack \
    --query_attack_target_language Python \
    --query_attack_use_icl true \
    --model gpt-4o \
    --samples 2
```

### Wiki-Text-Infilling (Academic framing)
```bash
python examples/universal_attack.py \
    --attack_name wiki_text_infilling \
    --wiki_variant sw \
    --model gpt-4o \
    --samples 2
```

## Multi-Provider Examples

### OpenAI Models
```bash
python examples/universal_attack.py \
    --attack_name abj \
    --model gpt-4o \
    --provider openai \
    --samples 3
```

### Anthropic Models
```bash
python examples/universal_attack.py \
    --attack_name query_attack \
    --model claude-3-sonnet-20240229 \
    --provider anthropic \
    --samples 2
```

### Azure OpenAI
```bash
python examples/universal_attack.py \
    --attack_name abj \
    --model gpt-4o \
    --provider azure \
    --samples 2
```

### Local Models (vLLM)
```bash
python examples/universal_attack.py \
    --attack_name query_attack \
    --model llama-2-7b-chat \
    --provider local \
    --samples 2
```

## Dataset Examples

### Built-in Datasets
```bash
# JBB harmful behaviors
python examples/universal_attack.py \
    --attack_name abj \
    --dataset jbb-harmful \
    --samples 10
```

### Custom Datasets
```bash
python examples/universal_attack.py \
    --attack_name query_attack \
    --dataset custom \
    --dataset_path data/my_prompts.json \
    --samples 20
```

### Dataset Sampling
```bash
python examples/universal_attack.py \
    --attack_name abj \
    --samples 50 \
    --start_index 100 \
    --seed 42
```

## Advanced Examples

### Custom Evaluation
```bash
python examples/universal_attack.py \
    --attack_name abj \
    --eval_model claude-3-sonnet-20240229 \
    --eval_provider anthropic \
    --model gpt-4o \
    --samples 5
```

### Batch Processing
```bash
python examples/universal_attack.py \
    --attack_name query_attack \
    --samples 100 \
    --output_dir batch_results/ \
    --seed 42
```

### Research Workflows
```bash
#!/bin/bash
# compare_attacks.sh

MODELS=("gpt-4o" "claude-3-sonnet-20240229")
ATTACKS=("abj" "mousetrap" "query_attack" "wiki_text_infilling")
DATASETS=("jbb-harmful")

for model in "${MODELS[@]}"; do
  for attack in "${ATTACKS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
      python examples/universal_attack.py \
        --attack_name $attack \
        --model $model \
        --dataset $dataset \
        --samples 50 \
        --output_dir "study_results/${model}/${attack}/${dataset}/"
    done
  done
done
```

## Debugging Examples

### Verbose Output
```bash
python examples/universal_attack.py \
    --attack_name abj \
    --model gpt-4o \
    --samples 1 \
    --verbose
```

### Getting Help
```bash
# Show all options
python examples/universal_attack.py --help

# Attack-specific help
python examples/universal_attack.py --attack_name abj --help
```

---

*For detailed CLI documentation, see [CLI_REFERENCE.md](CLI_REFERENCE.md). For development guides, see [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md).* 
