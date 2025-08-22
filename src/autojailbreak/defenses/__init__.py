"""
Defense implementations for the AutoJailbreak framework.
"""

from typing import Dict, Type, Optional, Any, List
from abc import ABC, abstractmethod

# Registry for defense implementations
_DEFENSE_REGISTRY: Dict[str, Type["BaseDefense"]] = {}


class BaseDefense(ABC):
    """Base class for defense implementations."""
    
    name: str = "base"
    description: str = "Base defense class"
    
    def __init__(self, **kwargs):
        """Initialize defense with parameters."""
        self.params = kwargs
    
    @abstractmethod
    def apply(self, prompt: str, **kwargs) -> str:
        """
        Apply the defense to a prompt.
        
        Args:
            prompt: The prompt to defend
            **kwargs: Additional parameters
            
        Returns:
            The defended prompt
        """
        pass
    
    @abstractmethod
    def process_response(self, response: str, **kwargs) -> str:
        """
        Process the model's response with the defense.
        
        Args:
            response: The model's response
            **kwargs: Additional parameters
            
        Returns:
            The processed response
        """
        pass
    
    @classmethod
    def from_config(cls, config: dict) -> "BaseDefense":
        """Create a defense instance from a configuration dictionary."""
        return cls(**config)


def register_defense(name: str):
    """Decorator to register a new defense implementation."""
    def _register(cls):
        cls.name = name
        _DEFENSE_REGISTRY[name] = cls
        return cls
    return _register


def get_defense(name: str, **kwargs) -> BaseDefense:
    """
    Get a defense implementation by name.
    
    Args:
        name: Name of the defense
        **kwargs: Parameters to pass to the defense constructor
        
    Returns:
        An instance of the defense
    """
    if name not in _DEFENSE_REGISTRY:
        raise ValueError(f"Defense '{name}' not found. Available defenses: {list(_DEFENSE_REGISTRY.keys())}")
    
    return _DEFENSE_REGISTRY[name](**kwargs)


def list_defenses() -> Dict[str, str]:
    """List all registered defenses with their descriptions."""
    return {name: cls.description for name, cls in _DEFENSE_REGISTRY.items()} 