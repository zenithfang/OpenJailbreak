# Contributing to OpenJailbreak

Welcome to OpenJailbreak! We appreciate your interest in contributing to this comprehensive framework for LLM jailbreak research. This guide will help you get started.

## Getting Started

### Prerequisites
- Python 3.8+
- Git
- API keys for at least one LLM provider (OpenAI, Anthropic, etc.)

### Development Setup

Follow the detailed setup instructions in [SETUP.md](SETUP.md), then:

1. **Fork and Clone**
   ```bash
   git fork https://github.com/zenithfang/OpenJailbreak.git
   git clone https://github.com/YOUR_USERNAME/OpenJailbreak.git
   cd OpenJailbreak
   ```

2. **Install with Development Dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

3. **Verify Setup**
   ```bash
   python examples/universal_attack.py --attack_name simple_override --samples 1
   ```

## Contributing Areas

### 🎯 **Adding New Components**

For detailed implementation guides, see the [Developer Guide](docs/DEVELOPER_GUIDE.md):

- **Adding New Attacks**: Complete guide with examples and requirements
- **Adding Defense Mechanisms**: Defense implementation and integration
- **Adding Evaluation Methods**: Custom evaluator development
- **Adding Model Providers**: LLM provider integration

**Quick Reference:**
- Attacks: Inherit from `ModernBaseAttack` with NAME, PAPER, PARAMETERS
- Defenses: Implement `BaseDefense` interface  
- Evaluators: Extend `JailbreakEvaluator` class
- Place files in appropriate `src/autojailbreak/` subdirectories

## Code Style Guidelines

### Python Style
- Follow PEP 8 with line length limit of 100 characters
- Use type hints for all function parameters and return values
- Include docstrings for all classes and public methods
- Use descriptive variable names

### Formatting Tools
```bash
# Format code
black src/ tests/ examples/
isort src/ tests/ examples/

# Lint code
ruff check src/ tests/ examples/
```

### Documentation
- Include comprehensive docstrings
- Add inline comments for complex logic
- Update CLI_REFERENCE.md for new parameters
- Update ATTACK_CONFIG.md for new attack types

## Testing Requirements

### Required Tests
- **Unit Tests**: Test individual attack components
- **Integration Tests**: Test attack execution pipeline
- **Parameter Tests**: Validate parameter handling
- **Regression Tests**: Ensure existing functionality works

### Test Structure
```bash
tests/
├── unit/
│   └── test_your_attack.py
├── integration/
│   └── test_your_attack_integration.py
└── fixtures/
    └── your_test_data.py
```

### Example Test
```python
import pytest
from src.autojailbreak.attacks.your_attack import YourNewAttack

class TestYourNewAttack:
    def test_attack_generation(self):
        attack = YourNewAttack()
        result = attack.generate_attack("test prompt", "test goal", "test target")
        assert isinstance(result, str)
        assert len(result) > 0
```

## Submission Process

### Branch Naming
- Feature: `feature/attack-name` or `feature/description`
- Bug fix: `fix/issue-description`
- Documentation: `docs/topic`

### Commit Messages
Use conventional commits format:
```
type(scope): description

feat(attacks): add new roleplay jailbreak attack
fix(evaluation): handle empty responses in string matcher
docs(readme): update installation instructions
test(attacks): add comprehensive tests for PAIR attack
```

### Pull Request Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-attack-name
   ```

2. **Implement Changes**
   - Write code following style guidelines
   - Add comprehensive tests
   - Update documentation if needed

3. **Test Locally**
   ```bash
   pytest tests/
   black --check .
   ruff check .
   ```

4. **Submit Pull Request**
   - Use clear, descriptive title
   - Fill out PR template completely
   - Link related issues
   - Request review from maintainers

### PR Requirements Checklist
- [ ] Code follows style guidelines
- [ ] Tests added and passing
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
- [ ] No hardcoded credentials or sensitive data
- [ ] Backwards compatibility maintained

## Research Ethics

When contributing jailbreak attacks:

### Responsible Disclosure
- Focus on research and defense improvement
- Do not include attacks targeting specific real-world systems
- Ensure attacks are for research purposes only
- Include appropriate disclaimers in attack documentation

### Academic Standards
- Always cite original research papers
- Include accurate paper references in PAPER attribute
- Respect intellectual property and licenses
- Give proper attribution to attack authors

### Safety Considerations
- Test attacks responsibly
- Do not use for malicious purposes
- Consider potential misuse when documenting
- Include warnings for potentially harmful attacks

## Getting Help

### Communication Channels
- **Issues**: Report bugs and request features
- **Discussions**: Ask questions and share ideas
- **Discord**: [Add Discord link if available]

### Documentation
- **README.md**: Overview and quick start
- **docs/CLI_REFERENCE.md**: Complete CLI documentation
- **docs/ATTACK_CONFIG.md**: Attack configuration guide

### Code Review Process
- All PRs require review from maintainers
- Feedback focuses on code quality, safety, and documentation
- Be patient and responsive to review comments
- Maintain respectful communication

## Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes for major contributions
- Academic papers that utilize contributed attacks (with permission)

Thank you for contributing to OpenJailbreak and advancing LLM safety research!

---

*For questions about this contributing guide, please open an issue or reach out to the maintainers.*
