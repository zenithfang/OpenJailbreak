"""Enhanced base classes for modular attack system."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Callable
import argparse
import logging

logger = logging.getLogger(__name__)


class AttackParameter:
    """Represents a configurable attack parameter."""
    
    def __init__(
        self,
        name: str,
        param_type: type,
        default: Any = None,
        description: str = "",
        cli_arg: Optional[str] = None,
        choices: Optional[List[Any]] = None,
        required: bool = False,
        validator: Optional[Callable[[Any], bool]] = None
    ):
        self.name = name
        self.type = param_type
        self.default = default
        self.description = description
        self.cli_arg = cli_arg
        self.choices = choices
        self.required = required
        self.validator = validator
    
    def validate(self, value: Any) -> bool:
        """Validate a parameter value."""
        if self.choices and value not in self.choices:
            return False
        if self.validator and not self.validator(value):
            return False
        return True
    
    def to_argparse_kwargs(self) -> Dict[str, Any]:
        """Convert to argparse.add_argument kwargs."""
        kwargs = {
            "type": self.type,
            "help": self.description,
            "default": self.default
        }
        if self.choices:
            kwargs["choices"] = self.choices
        if self.required:
            kwargs["required"] = True
        return {k: v for k, v in kwargs.items() if v is not None}


class ModernBaseAttack(ABC):
    """Enhanced base class for all attacks with embedded configuration."""
    
    # Class-level configuration - must be overridden by subclasses
    NAME: str = ""
    PAPER: str = ""
    PARAMETERS: Dict[str, AttackParameter] = {}
    
    def __init__(self, args=None, **kwargs):
        """Initialize attack with parameters."""
        self.args = args
        self.kwargs = kwargs
        self._validate_configuration()
    
    def _validate_configuration(self):
        """Validate the attack configuration."""
        required_fields = ["NAME"]
        for field in required_fields:
            if not getattr(self, field):
                raise ValueError(f"Attack {self.__class__.__name__} missing required field: {field}")
    
    @abstractmethod
    def generate_attack(self, prompt: str, goal: str, target: str, **kwargs) -> str:
        """Generate an attack string."""
        pass
    
    def get_parameter_value(self, param_name: str) -> Any:
        """Get a parameter value from args, kwargs, or default."""
        if param_name not in self.PARAMETERS:
            raise ValueError(f"Unknown parameter: {param_name}")
        
        param = self.PARAMETERS[param_name]
        
        # Check CLI args first
        if self.args and hasattr(self.args, param_name):
            value = getattr(self.args, param_name)
            if value is not None:
                return value
        
        # Check kwargs
        if param_name in self.kwargs:
            return self.kwargs[param_name]
        
        # Return default
        return param.default
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Return the complete attack configuration."""
        return {
            "name": cls.NAME,
            "paper": cls.PAPER,
            "parameters": {name: {
                "type": param.type.__name__,
                "default": param.default,
                "description": param.description,
                "cli_arg": param.cli_arg,
                "choices": param.choices,
                "required": param.required
            } for name, param in cls.PARAMETERS.items()}
        }
    
    @classmethod
    def get_cli_arguments(cls) -> List[Dict[str, Any]]:
        """Return CLI argument specifications for this attack."""
        args = []
        for param_name, param in cls.PARAMETERS.items():
            if param.cli_arg:
                args.append({
                    "name": param.cli_arg,
                    "kwargs": param.to_argparse_kwargs(),
                    "param_name": param_name
                })
        return args


# All attacks now inherit directly from ModernBaseAttack with minimal required attributes.