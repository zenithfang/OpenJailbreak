# OpenJailbreak

A comprehensive framework for automatic LLM jailbreak research with unified command-line interface and flattened architecture.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![GitHub](https://img.shields.io/badge/GitHub-zenithfang%2FOpenJailbreak-blue)](https://github.com/zenithfang/OpenJailbreak)

## Overview

OpenJailbreak provides a unified platform for researching and evaluating jailbreak attacks against Large Language Models (LLMs). The framework uses a **flattened registry-based architecture** with auto-discovery and embedded configuration for simplified attack development and usage. This repository underpins our systematic red‑teaming methodology.

**Key Features:**
- **Flattened Architecture**: Registry-based system with auto-discovery of attacks
- **Embedded Configuration**: Self-documenting attacks with built-in parameter definitions
- **Universal Command-Line Interface**: Single script with automatic CLI generation
- **Multiple Model Support**: OpenAI, Anthropic, Azure, Together AI, and local models
- **Standardized Evaluation**: Automated assessment with multiple evaluation methods
- **Research-Ready**: Reproducible experiments with comprehensive result tracking

## Quick Start

```bash
# Install
pip install -e .

# Set up API key (see SETUP.md for all providers)
export OPENAI_API_KEY="your_key_here"

# Run your first attack (ABJ - Chain-of-Thought exploit)
python examples/universal_attack.py \
    --attack_name abj \
    --model gpt-4o \
    --samples 3
```

## Available Attacks

| Attack | Description | Example |
|--------|-------------|---------|
| `abj` | Analyzing-based Jailbreak (CoT exploitation) | `--attack_name abj_attack` |
| `mousetrap` | Iterative chaos; reward hacking via decoding focus | `--attack_name mousetrap` |
| `query_attack` | Structured/C-like query manipulation | `--attack_name query_attack` |
| `wiki_text_infilling` | Academic framing with [MASK] infilling | `--attack_name wiki_text_infilling` |

```bash
# Get help for a specific attack
python examples/universal_attack.py --attack_name abj --help
```

## Reproduction Scripts

Reproduce the four core findings described in the writeup using ready-to-run scripts:

```bash
./examples/scripts/test_abj_attack.sh
./examples/scripts/test_mousetrap.sh
./examples/scripts/test_query_attack.sh
./examples/scripts/test_wiki_text_infilling.sh
```

See details and context in the **[Kaggle Writeup](kaggle_writeup.md)**.

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

# Get help for specific attack
python examples/universal_attack.py --attack_name abj --help
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