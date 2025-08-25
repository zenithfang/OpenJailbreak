# OpenJailbreak v1.0.0 🚀

## 🎯 Major Release: Comprehensive LLM Jailbreak Research Framework

OpenJailbreak v1.0.0 represents a complete, production-ready framework for automated LLM jailbreak research, featuring cutting-edge attack implementations and comprehensive research findings from systematic red-teaming efforts.

## 🔬 Research Highlights

### Key Findings from gpt-oss-20b Red-Teaming

Our systematic research uncovered critical vulnerabilities in LLM safety mechanisms:

**🔍 QueryAttack - Structured Language Exploitation**
- **Mechanism**: Structured, code-like formats manipulate rule-following and parsing
- **Critical Finding**: Models misclassify harmful content as benign under structured framing
- **Example**: C-like templates with ICL formatting to bypass safety filters

**🎭 Mousetrap - Chaos Transformation Attack**
- **Novel Approach**: Multiple encoding layers (Caesar cipher, word reversal, ASCII conversion)
- **Mechanism**: Models prioritize "solving" technical puzzles over safety evaluation

**🎪 Additional Breakthrough Attacks**
- **ABJ (Analyzing-based Jailbreak)**: CoT exploitation via data transformation and iterative toxicity adjustment
- **Wiki-Text-Infilling**: Academic framing with [MASK] infilling and wiki-context generation

## 🏗️ Framework Features

### Core Architecture
```
┌─────────────────────────────────────────┐
│          OpenJailbreak Core             │
├─────────────────────────────────────────┤
│  Attack Registry │ Auto-Discovery       │
│  ┌─────────────┐ │ ┌────────────────┐  │
│  │ QueryAttack │ │ │ Mousetrap      │  │
│  │ ABJ Attack  │ │ │ WikiInfilling  │  │
│  └─────────────┘ │ └────────────────┘  │
├─────────────────────────────────────────┤
│         LLM Provider Layer              │
│  OpenAI │ Anthropic │ Azure │ Local    │
├─────────────────────────────────────────┤
│       Evaluation Framework              │
│  String Match │ LLM Judge │ Toxicity   │
└─────────────────────────────────────────┘
```

### 🎯 Key Features
- **Auto-Discovery Registry**: Zero-configuration attack loading
- **Universal CLI**: Single entry point with automatic parameter generation
- **Multi-Provider Support**: OpenAI, Anthropic, Azure, local models via vLLM
- **Embedded Configuration**: Self-documenting attacks with built-in parameters
- **Evaluation Framework**: Multiple assessment methods (string matching, LLM judge, toxicity scoring)
- **Reproducible Research**: Deterministic seeds and comprehensive logging

## 📦 What's Included

### Attack Implementations
- **QueryAttack** - Structured language exploitation (arXiv:2502.09723)
- **Mousetrap** - Multi-layer chaos transformations  
- **ABJ Attack** - Assumed behavior jailbreaking
- **Wiki-Text Infilling** - Text completion exploitation

### Evaluation Methods
- **String Matching**: Fast keyword-based detection
- **LLM-as-Judge**: Model-based safety assessment  
- **Domain-Specific Evaluators**: GSM8K (math), WMDP (knowledge)
- **Toxicity Scoring**: Automated harmful content detection

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/zenithfang/OpenJailbreak.git
cd OpenJailbreak

# Install the framework
pip install -e .

# Set up API key
export OPENAI_API_KEY="your_key_here"
```

### Run Your First Attack
```bash
# QueryAttack example
python examples/universal_attack.py \
    --attack_name query_attack \
    --model gpt-4o \
    --samples 5 \
    --query_attack_target_language C
```

### Research Usage (Programmatic)
```python
import src.autojailbreak as ajb

attack = ajb.AttackFactory.create_attack("abj")
model = ajb.LLMLiteLLM.from_config(model_name="gpt-4o", provider="openai")

attack_prompt = attack.generate_attack(
    prompt="Your test prompt",
    goal="research_safety",
    target="target_model"
)
response = model.query(attack_prompt)
```

## 🔒 Security & Ethics

- **Responsible Disclosure**: All findings reported through appropriate channels
- **Research Focus**: Framework designed for academic and safety research
- **Ethical Guidelines**: Comprehensive documentation on responsible usage
- **No Malicious Intent**: All tools designed to improve LLM safety

## 🎓 Research Impact

### Publications & Citations
- Based on and extending work from arXiv:2502.09723 (QueryAttack)
- Novel contributions in chaos transformation attacks
- Systematic analysis of Chain-of-Thought safety failures

### Community Contributions
- Open-source framework accelerating research collaboration
- Standardized evaluation methods for consistent comparison
- Comprehensive documentation enabling research reproducibility

## 🛣️ Roadmap

### Upcoming Features
- **Cross-Model Transferability**: Testing attack generalization
- **Automated Defense Development**: Dynamic countermeasure generation  
- **Advanced Evaluation Metrics**: Quantitative safety assessment
- **Real-Time Attack Detection**: Live monitoring capabilities

## 🤝 Contributing

OpenJailbreak is open-source and welcomes community contributions:

- **Add New Attacks**: Inherit from `ModernBaseAttack` with 3 simple attributes
- **Extend Evaluators**: Implement domain-specific safety assessment
- **Improve Defenses**: Develop novel countermeasures
- **Enhance Documentation**: Help others understand and use the framework

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📚 Documentation

- **[📖 User Guide](docs/USER_GUIDE.md)** - Complete usage documentation
- **[🏗️ Developer Guide](docs/DEVELOPER_GUIDE.md)** - Framework extension guide  
- **[⌨️ CLI Reference](docs/CLI_REFERENCE.md)** - Command-line documentation
- **[📊 Examples](docs/EXAMPLES.md)** - Comprehensive usage examples
- **[🚀 Setup Guide](SETUP.md)** - Installation and configuration

## 🏆 Recognition

This release includes research findings that contribute to:
- **AI Safety**: Systematic vulnerability discovery
- **Red-Teaming Methodology**: Automated attack development
- **Community Tools**: Open-source framework for collaboration
- **Reproducible Research**: Standardized evaluation and reporting

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/zenithfang/OpenJailbreak/issues)
- **Discussions**: [GitHub Discussions](https://github.com/zenithfang/OpenJailbreak/discussions)  
- **Documentation**: Comprehensive guides in `docs/` directory
- **Examples**: Working examples in `examples/` directory

## 🙏 Acknowledgments

Special thanks to:
- The broader red-teaming community for shared insights and collaboration
- Researchers whose work we build upon and cite
- The AI safety community for emphasizing responsible research practices
- Competition organizers for creating platforms for collaborative safety research

---

**Download**: [Source Code (tar.gz)](https://github.com/zenithfang/OpenJailbreak/archive/v1.0.0.tar.gz) | [Source Code (zip)](https://github.com/zenithfang/OpenJailbreak/archive/v1.0.0.zip)

**Full Changelog**: https://github.com/zenithfang/OpenJailbreak/commits/v1.0.0

**Repository**: https://github.com/zenithfang/OpenJailbreak  
**License**: MIT  
**Version**: v1.0.0

*This release represents a comprehensive platform for LLM safety research. Use responsibly and contribute to making AI systems safer for everyone.*
