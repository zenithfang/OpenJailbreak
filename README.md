# OpenJailbreak

A comprehensive framework for automatic LLM jailbreak research with unified command-line interface and flattened architecture.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![GitHub](https://img.shields.io/badge/GitHub-zenithfang%2FOpenJailbreak-blue)](https://github.com/zenithfang/OpenJailbreak)

## Overview

OpenJailbreak provides a unified platform for researching and evaluating jailbreak attacks against Large Language Models (LLMs). The framework uses a **flattened registry-based architecture** with auto-discovery and embedded configuration for simplified attack development and usage.

**Key Features:**
- **Flattened Architecture**: Registry-based system with auto-discovery of attacks
- **Embedded Configuration**: Self-documenting attacks with built-in parameter definitions
- **Universal Command-Line Interface**: Single script with automatic CLI generation
- **Multiple Model Support**: OpenAI, Anthropic, Azure, Together AI, and local models
- **Built-in Defenses**: Test robustness with integrated defense mechanisms
- **Standardized Evaluation**: Automated assessment with multiple evaluation methods
- **Research-Ready**: Reproducible experiments with comprehensive result tracking

## Quick Start

```bash
# Install
pip install -e .

# Set up API key (see SETUP.md for all providers)
export OPENAI_API_KEY="your_key_here"

# Run your first attack
python examples/universal_attack.py \
    --attack_name simple_override \
    --model gpt-4o \
    --samples 3
```

## Available Attacks

| Attack | Description | Example |
|--------|-------------|---------|
| `simple_override` | Basic system prompt override | `--attack_name simple_override` |
| `role_assignment` | Assign harmful role to model | `--attack_name role_assignment` |
| `many_shot` | Many-shot in-context learning | `--attack_name many_shot` |
| `pair` | Prompt Automatic Iterative Refinement | `--attack_name pair` |

```bash
# View all available attacks
python examples/universal_attack.py --list_attacks
```

## Documentation

- **[🚀 Setup Guide](SETUP.md)** - Installation, API keys, and configuration
- **[📖 User Guide](docs/USER_GUIDE.md)** - End-user documentation and tutorials
- **[⌨️ CLI Reference](docs/CLI_REFERENCE.md)** - Complete command-line documentation
- **[🏗️ Developer Guide](docs/DEVELOPER_GUIDE.md)** - Creating attacks, defenses, and evaluators
- **[📚 Examples](docs/EXAMPLES.md)** - Comprehensive usage examples
- **[🤝 Contributing](CONTRIBUTING.md)** - How to contribute to the project

## Quick Help

```bash
# Show all available options
python examples/universal_attack.py --help

# List available attacks
python examples/universal_attack.py --list_attacks

# Get help for specific attack
python examples/universal_attack.py --attack_name many_shot --help
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

**Repository**: [github.com/zenithfang/OpenJailbreak](https://github.com/zenithfang/OpenJailbreak)

The flattened architecture makes adding new attacks straightforward:
1. Create a class inheriting from `ModernBaseAttack`
2. Define NAME, PAPER, and PARAMETERS
3. Framework auto-discovers and integrates it

## Research Applications

OpenJailbreak is used in research on AI safety, adversarial prompt engineering, defense evaluation, and comparative jailbreak analysis.

## License

MIT License - see [LICENSE](LICENSE) file for details. 