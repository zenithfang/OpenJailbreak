# Developer Guide

Comprehensive guide for developing new attacks, defenses, and evaluators in OpenJailbreak.

## Architecture Overview

OpenJailbreak uses a **flattened registry-based architecture** with automatic component discovery. The system eliminates complex hierarchical groupings in favor of simple, self-documenting components.

### Core Design Principles
- **Auto-Discovery**: No manual registration required
- **Embedded Configuration**: Configuration lives with implementation
- **Minimal Boilerplate**: Only 3 required attributes for attacks
- **Type Safety**: Built-in parameter validation
- **CLI Integration**: Automatic command-line generation

### System Components
```
AttackRegistry ←→ ModernBaseAttack (discovers and instantiates)
    ↓
LLM Provider Layer ←→ AttackExecution 
    ↓
Defense Layer (optional) ←→ Response Processing
    ↓  
Evaluation Framework ←→ JailbreakEvaluator
    ↓
Results Output ←→ Configuration Management
```

## Creating New Attacks

### Attack Structure (v2.0)
Each attack requires only **3 essential attributes**:

```python
from src.autojailbreak.attacks.base import ModernBaseAttack, AttackParameter

class YourNewAttack(ModernBaseAttack):
    """Brief description of your attack technique."""
    
    NAME = "your_attack_name"
    PAPER = "Author et al. - Paper Title (Conference Year)"
    
    PARAMETERS = {
        "your_param": AttackParameter(
            name="your_param",
            param_type=str,
            default="default_value",
            description="Parameter description for CLI help",
            cli_arg="--your_param"
        )
    }
    
    def generate_attack(self, prompt: str, goal: str, target: str, **kwargs) -> str:
        """Generate the jailbreak attack string."""
        param_value = self.get_parameter_value("your_param")
        # Your attack logic here
        return modified_prompt
```

### Essential Attributes

#### **NAME** (Required)
- Unique identifier for the attack
- Used for CLI selection: `--attack_name your_name`
- Use lowercase with underscores, e.g., `abj`, `mousetrap`

#### **PAPER** (Required)
- Research paper reference for academic context
- Format: "Author et al. - Full Title (Conference Year)"
- Example: "Zou et al. - Universal and Transferable Adversarial Attacks (NeurIPS 2023)"

#### **PARAMETERS** (Required)
- Dictionary of configurable parameters
- Each parameter is an `AttackParameter` object
- Enables automatic CLI generation and validation

### Parameter System

#### AttackParameter Definition
```python
AttackParameter(
    name="parameter_name",           # Internal parameter name
    param_type=str,                  # Parameter type (str, int, bool, float)
    default="default_value",         # Default value
    description="Parameter description",  # Help text for CLI
    cli_arg="--cli_argument",        # Command-line argument name
    choices=["option1", "option2"],  # Valid choices (optional)
    required=False,                  # Whether parameter is required
    validator=custom_validator_func  # Custom validation function (optional)
)
```

#### Parameter Types
- **str**: Text parameters, file paths, comma-separated lists
- **int**: Counts, limits, indices
- **bool**: Boolean flags (use negated flags for False defaults)
- **float**: Temperatures, thresholds, percentages

#### Parameter Access
```python
def generate_attack(self, prompt: str, goal: str, target: str, **kwargs) -> str:
    # Get parameter value (checks CLI args, then kwargs, then default)
    param_value = self.get_parameter_value("parameter_name")
    
    # Use in attack logic
    if param_value == "special_mode":
        return self._special_attack_logic(prompt)
    return self._default_attack_logic(prompt, param_value)
```

### Attack Implementation Examples

#### Simple Parameter Attack
```python
class ExampleAttack(ModernBaseAttack):
    """Minimal example to demonstrate parameters."""

    NAME = "example_attack"
    PAPER = "Your Paper Title"

    PARAMETERS = {
        "example_param": AttackParameter(
            name="example_param",
            param_type=str,
            default="default_value",
            description="Example parameter",
            cli_arg="--example_param"
        )
    }

    def generate_attack(self, prompt: str, goal: str, target: str, **kwargs) -> str:
        value = self.get_parameter_value("example_param")
        return f"[example:{value}] {prompt}"
```

#### Complex Parameter Attack
```python
class AbjLikeAttack(ModernBaseAttack):
    """Example with multiple parameters (pattern similar to ABJ)."""

    NAME = "abj_like"
    PAPER = "Method Reference"

    PARAMETERS = {
        "max_rounds": AttackParameter(name="max_rounds", param_type=int, default=1, cli_arg="--max_rounds"),
        "judge_model": AttackParameter(name="judge_model", param_type=str, default="gpt-4o", cli_arg="--judge_model")
    }
```

### File Organization
Place attack files in `src/autojailbreak/attacks/`:
```
src/autojailbreak/attacks/
├── __init__.py
├── base.py              # Base classes (don't modify)
├── registry.py          # Registry system (don't modify)
├── your_attack.py       # Your new attack here
├── abj_attack.py
├── mousetrap.py
├── query_attack.py
└── wiki_text_infilling.py
```

### Testing Your Attack
```python
# tests/unit/test_your_attack.py
import pytest
from src.autojailbreak.attacks.your_attack import YourNewAttack

class TestYourNewAttack:
    def test_attack_generation(self):
        attack = YourNewAttack()
        result = attack.generate_attack("test prompt", "test goal", "test target")
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_parameter_handling(self):
        attack = YourNewAttack()
        # Test parameter access
        param_value = attack.get_parameter_value("your_param")
        assert param_value is not None
```

## Creating Defense Mechanisms

### Defense Interface
```python
from src.autojailbreak.defenses.base import BaseDefense

class YourDefense(BaseDefense):
    """Description of your defense mechanism."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize defense-specific parameters
        self.threshold = kwargs.get('threshold', 0.5)
    
    def apply(self, response: str) -> str:
        """Apply defense to model response."""
        # Your defense logic here
        if self._is_suspicious(response):
            return "I cannot provide that information."
        return response
    
    def _is_suspicious(self, response: str) -> bool:
        """Helper method to detect suspicious content."""
        # Your detection logic
        return False
```

### Defense Registration
Place defense files in `src/autojailbreak/defenses/`:
```
src/autojailbreak/defenses/
├── __init__.py
├── base.py              # Base defense class
├── your_defense.py      # Your new defense
├── smoothllm.py         # Example defenses
└── perplexity.py
```

## Creating Evaluation Methods

### Evaluator Interface
```python
from src.autojailbreak.evaluation.base import JailbreakEvaluator

class YourEvaluator(JailbreakEvaluator):
    """Description of your evaluation method."""
    
    def __init__(self, **kwargs):
        super().__init__(name="your_evaluator", **kwargs)
        # Initialize evaluator-specific parameters
    
    def evaluate(self, prompt: str, response: str, **kwargs) -> dict:
        """Evaluate attack success."""
        # Your evaluation logic
        success = self._determine_success(response)
        confidence = self._calculate_confidence(response)
        
        return {
            "success": success,
            "confidence": confidence,
            "explanation": "Detailed explanation of evaluation"
        }
    
    def _determine_success(self, response: str) -> bool:
        """Determine if attack was successful."""
        # Your success detection logic
        return False
```

### Domain-Specific Evaluators
```python
# For specialized domains like math or knowledge
class MathEvaluator(JailbreakEvaluator):
    """Evaluator for mathematical reasoning attacks."""
    
    def __init__(self, **kwargs):
        super().__init__(name="math", **kwargs)
        self.numerical_tolerance = kwargs.get('numerical_tolerance', 1e-6)
    
    def evaluate(self, prompt: str, response: str, **kwargs) -> dict:
        expected_answer = kwargs.get('expected_answer')
        extracted_answer = self._extract_numerical_answer(response)
        
        success = abs(extracted_answer - expected_answer) < self.numerical_tolerance
        
        return {
            "success": success,
            "confidence": 1.0 if success else 0.0,
            "extracted_answer": extracted_answer,
            "expected_answer": expected_answer
        }
```

## Advanced Development

### Custom Validators
```python
def validate_positive_integer(value):
    """Validate that parameter is a positive integer."""
    return isinstance(value, int) and value > 0

def validate_file_exists(path):
    """Validate that file path exists."""
    return path is None or os.path.exists(path)

def validate_language_list(languages):
    """Validate comma-separated language list."""
    supported_languages = ["English", "Spanish", "French", "German", "Chinese"]
    lang_list = [lang.strip() for lang in languages.split(",")]
    return all(lang in supported_languages for lang in lang_list)
```

### Error Handling
```python
def generate_attack(self, prompt: str, goal: str, target: str, **kwargs) -> str:
    try:
        param_value = self.get_parameter_value("critical_param")
        
        if not param_value:
            raise ValueError("Critical parameter cannot be empty")
        
        # Attack generation logic
        result = self._generate_with_param(prompt, param_value)
        
        if not result:
            logger.warning("Attack generation produced empty result")
            return prompt  # Fallback to original prompt
        
        return result
        
    except Exception as e:
        logger.error(f"Attack generation failed: {e}")
        raise
```

### Performance Optimization
```python
class OptimizedAttack(ModernBaseAttack):
    """Example of performance-optimized attack."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Pre-compute expensive operations
        self._template_cache = {}
        self._compiled_patterns = self._compile_patterns()
    
    def generate_attack(self, prompt: str, goal: str, target: str, **kwargs) -> str:
        # Use cached computations
        template_key = self._get_template_key(prompt)
        if template_key in self._template_cache:
            return self._template_cache[template_key].format(prompt=prompt)
        
        # Generate and cache
        result = self._expensive_generation(prompt)
        self._template_cache[template_key] = result
        return result
```

## Legacy Migration Guide

### Removing Deprecated Attributes
If updating old attacks, remove these deprecated attributes:

```python
# ❌ Remove these (no longer used)
TYPE = "obfuscation"           
METHOD = "translation"         
SUBMETHOD = "chain"            
DESCRIPTION = "Attack description"

# ✅ Keep these (essential)
NAME = "attack_name"
PAPER = "Research paper reference"
PARAMETERS = {...}
```

### Benefits of Minimalist Design
- **28 lines of dead code removed** across all attacks
- **Faster loading** due to simplified structure
- **Better IDE support** with cleaner interfaces
- **Self-documenting** through Python docstrings

## Best Practices

### Code Quality
- Use type hints for all parameters and returns
- Include comprehensive docstrings
- Follow PEP 8 style guidelines (100 character line limit)
- Add unit tests for all new components

### Documentation
- Use descriptive attack and parameter names
- Include paper references for academic context
- Add inline comments for complex logic
- Update user documentation when adding new features

### Security Considerations
- Never include hardcoded credentials
- Validate all user inputs
- Consider potential misuse of attack techniques
- Include appropriate warnings for dangerous attacks

### Testing Requirements
- Unit tests for attack generation logic
- Integration tests for full attack pipeline
- Parameter validation tests
- Performance benchmarks for complex attacks

---

*For setup instructions, see [SETUP.md](../SETUP.md). For usage examples, see [EXAMPLES.md](EXAMPLES.md).*
 
## Composability for Series Runner

### Contract for Series-Compatible Attacks
- Implement `generate_attack(self, prompt: str, goal: str, target: str, **kwargs)`
- Return type: `str` or `List[str]` (the series runner accepts both; lists use the first item by default)
- Maintain the three required attributes so parameters are exposed via CLI:
  - `NAME` (unique, lowercase with underscores)
  - `PAPER` (reference string)
  - `PARAMETERS` (dict of `AttackParameter` entries)

### Making Parameters CLI-Exposable
- Ensure each parameter includes a `cli_arg` so it can be configured when chained
- Validate inputs with `choices`, `required`, or `validator` where appropriate

### Series Configuration Schema
Minimal JSON used by the series runner:
```json
{
  "attack_chain": [
    { "name": "past_tense_attack", "parameters": {} },
    { "name": "translate_chain", "parameters": {} }
  ],
  "metadata": {
    "created_by": "AutoJailbreak Series Attack Framework",
    "version": "1.0"
  }
}
```

- Example config: `assets/series_configs/series_simple.json`
- Each `parameters` block is merged with CLI args at execution time

### Local Testing Tips
- Use a small sample size (e.g., `--samples 2`) during development
- Ensure UTF‑8/Unicode locale for translation-based components
- Combine with `--verbose` and `--resume` to iterate quickly