# User Guide

Complete guide for using OpenJailbreak to conduct LLM jailbreak research.

## Getting Started

### Prerequisites
Before using OpenJailbreak, ensure you have:
- Python 3.8+ installed
- API access to at least one LLM provider
- Basic familiarity with command-line interfaces

### First Steps
1. **Install**: Follow the [Setup Guide](../SETUP.md)
2. **Configure**: Set up API keys for your chosen provider
3. **Test**: Run your first attack to verify everything works

## Working with Attacks

### Basic Attack Usage
```bash
# Run a basic attack
python examples/universal_attack.py \
    --attack_name abj \
    --model gpt-4o \
    --samples 5
```

### Understanding Attack Types

#### **Chain-of-Thought Exploitation**
- **Purpose**: Manipulate CoT reasoning as an internal channel
- **Example**: `abj`

#### **Reward Hacking**
- **Purpose**: Divert model objectives toward task completion over safety
- **Example**: `mousetrap`

### Customizing Attack Parameters
Each attack has configurable parameters. Use `--help` to see available options:

```bash
# See all options for specific attack
python examples/universal_attack.py --attack_name abj --help

# Common parameters
--samples 10           # Number of test cases
--model gpt-4o        # Target model
--provider openai     # Model provider
--verbose             # Detailed output
```

## Supported Models

### Cloud Providers

#### OpenAI
- **Models**: `gpt-4o`, `gpt-4-turbo`, `gpt-3.5-turbo`
- **Setup**: See [SETUP.md](../SETUP.md#openai)
- **Usage**: `--provider openai --model gpt-4o`

#### Anthropic
- **Models**: `claude-3-sonnet-20240229`, `claude-3-opus-20240229`, `claude-3-haiku-20240307`
- **Setup**: See [SETUP.md](../SETUP.md#anthropic-claude)
- **Usage**: `--provider anthropic --model claude-3-sonnet-20240229`

#### Azure OpenAI
- **Models**: All OpenAI models via Azure endpoints
- **Setup**: Requires additional AZURE_API_BASE configuration
- **Usage**: `--provider azure --model gpt-4o`

#### AWS Bedrock
- **Models**: `us.anthropic.claude-sonnet-4-20250514-v1:0` and similar
- **Setup**: AWS credentials required
- **Usage**: `--provider bedrock --model us.anthropic.claude-sonnet-4-20250514-v1:0`

#### Google Vertex AI
- **Models**: `gemini-2.5-pro`, `gemini-pro-vision`
- **Setup**: Google Cloud project and authentication
- **Usage**: `--provider vertex_ai --model gemini-2.5-pro`

#### Aliyun DashScope
- **Models**: All Aliyun models
- **Setup**: DashScope API key required
- **Usage**: `--provider aliyun --model model_name`

### Local Models
- **Framework**: Any model supported by vLLM
- **Setup**: `pip install -e ".[vllm]"`
- **Usage**: `--provider local --model your_model_name`

## Working with Datasets

### Built-in Datasets

#### JBB (JailbreakBench) Datasets
```bash
# Harmful behaviors dataset
--dataset jbb-harmful

# Benign behaviors dataset  
--dataset jbb-benign

# All behaviors combined
--dataset jbb-all
```

#### Research Benchmarks
```bash
# AdvBench dataset
--dataset advbench

# HarmBench dataset
--dataset harmbench

# HarmBench contextual subset
--dataset harmbench-contextual
```

### Custom Datasets

#### JSON Format
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

#### Usage
```bash
# Load custom dataset
python examples/universal_attack.py \
    --attack_name query_attack \
    --dataset custom \
    --dataset_path data/my_dataset.json
```

### Dataset Sampling
```bash
# Control sample size
--samples 10          # Test 10 examples
--start_index 5       # Start from index 5
--all_samples         # Use entire dataset
```

## Evaluation Methods

### Automatic Evaluation
OpenJailbreak provides several evaluation methods:

#### String Matching
- **Speed**: Fast
- **Accuracy**: Basic
- **Use**: Quick initial assessment

#### LLM-as-Judge
- **Speed**: Slow
- **Accuracy**: High
- **Use**: Nuanced evaluation
- **Configuration**: `--eval_model gpt-4o --eval_provider openai`

#### Domain-Specific Evaluators
- **GSM8K**: For mathematical reasoning attacks
- **WMDP**: For knowledge-based attacks
- **Custom**: Implement your own evaluators

### Evaluation Configuration
```bash
# Use different evaluation model
--eval_model claude-3-sonnet-20240229
--eval_provider anthropic

# Save detailed evaluation results
--save_attacks
--output_dir detailed_results/
```

## Output and Results

### Result Formats
OpenJailbreak outputs results in multiple formats:

#### Console Output
Real-time progress and summary statistics displayed during execution.

#### JSON Results
Detailed results saved to JSON files with:
- Attack success rates
- Individual response evaluations
- Timing and performance metrics
- Configuration used

### Result Organization
```bash
# Control output location
--output_dir experiments/my_research/
--output my_experiment_results.json

# Include attack prompts in output
--save_attacks
```

### Analyzing Results
```python
import json
import pandas as pd

# Load results
with open('results.json', 'r') as f:
    results = json.load(f)

# Convert to DataFrame for analysis
df = pd.DataFrame(results['individual_results'])
success_rate = df['evaluation_result'].mean()
```

## Advanced Usage

### Batch Operations
```bash
# Test multiple samples efficiently
python examples/universal_attack.py \
    --attack_name abj \
    --samples 100 \
    --output_dir batch_results/

# Use comprehensive testing script
python examples/scripts/test_comprehensive.py \
    --attack_name query_attack \
    --all_samples
```

### Research Workflows
```bash
# Compare multiple attacks
for attack in simple_override many_shot pair; do
    python examples/universal_attack.py \
        --attack_name $attack \
        --samples 50 \
        --output_dir comparison_study/
done

# Test across multiple models
python examples/scripts/test_comprehensive.py \
    --attack_name many_shot \
    --samples 20
```

### Performance Optimization
```bash
# Use parallel processing
export MAX_WORKERS=8

# Set appropriate timeouts
export TIMEOUT_SECONDS=120

# Enable caching
export DATASET_CACHE_DIR=".cache/datasets"
```

## Programmatic Usage

### Python API
```python
import src.autojailbreak as ajb

# Load attack
attack = ajb.AttackFactory.create_attack("abj")

# Load model
model = ajb.LLMLiteLLM.from_config(
    model_name="gpt-4o",
    provider="openai"
)

# Load dataset
dataset = ajb.read_dataset("jbb-harmful")

# Run evaluation (manual loop)
evaluator = ajb.JailbreakEvaluator()
for item in dataset.sample(5, 42):
    attack_prompt = attack.generate_attack(
        prompt=item["goal"],
        goal=item["goal"],
        target=item.get("target", "")
    )
    response = model.query(attack_prompt)
    result = evaluator({"question": item["goal"], "answer": response})
```

### Integration with Research Tools
```python
# Integration with existing research pipelines
from autojailbreak import AttackFactory, LLMLiteLLM

def run_experiment(attack_name, model_config, samples):
    attack = AttackFactory.create_attack(attack_name)
    model = LLMLiteLLM.from_config(**model_config)
    # Your research logic here
```

## Best Practices

### Research Ethics
- Use only for legitimate research purposes
- Follow institutional ethics guidelines
- Consider potential misuse when sharing results
- Include appropriate disclaimers in publications

### Reproducibility
- Always use `--seed` for deterministic results
- Save complete configuration with results
- Document model versions and provider settings
- Include framework version in research reports

### Performance
- Start with small sample sizes for testing
- Use `--verbose` only when debugging
- Consider rate limits when using cloud APIs
- Cache datasets locally for repeated experiments

---

*For setup and installation, see [SETUP.md](../SETUP.md). For contributing, see [CONTRIBUTING.md](../CONTRIBUTING.md).*
 
## Series/Compositional Attacks

### When to Use
- To test whether chaining simple transformations (e.g., tense shift + translation) is more effective than single-step attacks.

### How It Works
- The series runner executes a list of components sequentially. The output of step N becomes the input of step N+1. The final prompt is sent to the target model and evaluated automatically.

### Quick Start
```bash
python examples/universal_attack_series.py \
  --config_file assets/series_configs/series_simple.json \
  --model gpt-4o \
  --provider openai \
  --samples 3
```

Or specify the chain inline:
```bash
python examples/universal_attack_series.py \
  --attack_chain "past_tense_attack,translate_chain" \
  --model gpt-4o --provider openai
```

### Caveats
- Non‑ASCII text: translation-based chains require UTF‑8/Unicode-capable shells and fonts.
- API keys: you need keys for both the target model provider and the evaluation provider (e.g., OpenAI for `--eval_provider openai`).

### Reproduce the Compositional Finding
```bash
./examples/scripts/test_series_attack.sh
```