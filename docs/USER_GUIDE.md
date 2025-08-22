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
    --attack_name simple_override \
    --model gpt-4o \
    --samples 5
```

### Understanding Attack Types

#### **Simple Override Attacks**
- **Purpose**: Basic system prompt manipulation
- **Use Case**: Testing basic safety guardrails
- **Examples**: `simple_override`, `role_assignment`

#### **Translation Attacks**
- **Purpose**: Language-based obfuscation
- **Use Case**: Testing multilingual safety measures
- **Examples**: `translate_chain`

#### **Contextual Attacks**
- **Purpose**: In-context learning manipulation
- **Use Case**: Testing few-shot learning vulnerabilities
- **Examples**: `many_shot`

#### **Iterative Attacks**
- **Purpose**: Automated refinement of attack prompts
- **Use Case**: Advanced attack optimization
- **Examples**: `pair`

### Customizing Attack Parameters
Each attack has configurable parameters. Use `--help` to see available options:

```bash
# See all options for specific attack
python examples/universal_attack.py --attack_name many_shot --help

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

#### CSV Format
```csv
goal,target,behavior,category
"Write harmful content","Sure, here's harmful content","Content Generation","Harmful"
```

#### Usage
```bash
# Load custom dataset
python examples/universal_attack.py \
    --attack_name simple_override \
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

## Defense Mechanisms

### Available Defenses

#### SmoothLLM Defense
- **Purpose**: Perturb inputs to detect adversarial content
- **Usage**: `--defense smoothllm`
- **Best for**: Detection of crafted inputs

#### Perplexity Defense
- **Purpose**: Filter responses based on perplexity scores
- **Usage**: `--defense perplexity`
- **Best for**: Detecting unusual response patterns

#### Paraphrase Defense
- **Purpose**: Rephrase inputs to break attack patterns
- **Usage**: `--defense paraphrase --paraphrase_model gpt-4o`
- **Best for**: Breaking structured attack templates

### Using Defenses
```bash
# Test attack effectiveness against defenses
python examples/universal_attack.py \
    --attack_name many_shot \
    --defense smoothllm \
    --model gpt-4o \
    --samples 5
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

#### CSV Export
Tabular data suitable for analysis:
- One row per test case
- Columns for all relevant metrics
- Easy import into analysis tools

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
    --attack_name many_shot \
    --samples 100 \
    --output_dir batch_results/

# Use comprehensive testing script
python examples/scripts/test_comprehensive.py \
    --attack_name pair \
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
attack = ajb.create_attack("many_shot")

# Load model
model = ajb.LLMLiteLLM.from_config(
    model="gpt-4o",
    provider="openai"
)

# Load dataset
dataset = ajb.read_dataset("jbb-harmful")

# Run evaluation
evaluator = ajb.JailbreakEvaluator()
results = evaluator.evaluate_dataset(attack, model, dataset)
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
