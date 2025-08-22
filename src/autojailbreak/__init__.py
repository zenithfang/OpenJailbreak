"""
AutoJailbreak: An integrated framework for automatic LLM jailbreak techniques.
"""

__version__ = "0.1.0"

from .config import read_config
from .dataset import read_dataset
from .artifact import read_artifact, JailbreakInfo, ArtifactParameters
from .llm.base import BaseLLM
from .llm.litellm import LLMLiteLLM
from .llm.vllm import LLMvLLM
from .attacks import BaseAttack
from .defenses import BaseDefense, register_defense, get_defense
from .evaluation import JailbreakEvaluator, evaluate_jailbreak

# Import domain-specific evaluators if available
try:
    from .evaluation.wmdp_evaluator import WMDPEvaluator
    WMDP_EVALUATOR_AVAILABLE = True
except ImportError:
    WMDP_EVALUATOR_AVAILABLE = False

try:
    from .evaluation.gsm8k_evaluator import GSM8KEvaluator
    GSM8K_EVALUATOR_AVAILABLE = True
except ImportError:
    GSM8K_EVALUATOR_AVAILABLE = False

# Modular attack system is now the primary interface
# Individual attacks are auto-discovered through the registry

# Register all included defenses
from .defenses.smoothllm import SmoothLLM
from .defenses.perplexity import PerplexityFilter
from .defenses.retokenization import RetokenizationDefense
from .defenses.dictionary import DictionaryFilter

__all__ = [
    # Core
    "read_config", "read_dataset", "read_artifact", "JailbreakInfo", "ArtifactParameters",
    # LLM interfaces
    "BaseLLM", "LLMLiteLLM", "LLMvLLM",
    # Attacks
    "BaseAttack",
    # Defenses
    "BaseDefense", "register_defense", "get_defense",
    # Defense implementations
    "SmoothLLM", "PerplexityFilter", "RetokenizationDefense", "DictionaryFilter",
    # Evaluation
    "JailbreakEvaluator", "evaluate_jailbreak"
]

# Add domain-specific evaluators to exports if available
if WMDP_EVALUATOR_AVAILABLE:
    __all__.append("WMDPEvaluator")

if GSM8K_EVALUATOR_AVAILABLE:
    __all__.append("GSM8KEvaluator") 