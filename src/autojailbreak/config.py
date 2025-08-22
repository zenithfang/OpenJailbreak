"""
Configuration module for OpenJailbreak framework.
"""

import os
import json
import yaml
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union


@dataclass
class JailbreakConfig:
    """Configuration for jailbreak experiments."""
    
    # General settings
    output_dir: str = "results"
    log_dir: str = "logs"
    cache_dir: str = os.path.expanduser("~/.cache/openjailbreak")
    
    # Dataset settings
    dataset_name: str = "JBB-behaviors"
    dataset_path: Optional[str] = None
    
    # Model settings
    model_name: str = "vicuna-13b-v1.5"
    model_provider: str = "together"
    use_vllm: bool = False
    
    # Attack settings
    attack_type: str = "simple"
    attack_params: Dict[str, Any] = None
    
    # Defense settings
    defense_type: Optional[str] = None
    defense_params: Dict[str, Any] = None
    
    # Evaluation settings
    evaluation_method: str = "string_match"
    judge_model: Optional[str] = None
    
    # API keys and endpoints
    api_keys: Dict[str, str] = None
    endpoints: Dict[str, str] = None


def read_config(config_path: str) -> JailbreakConfig:
    """Read configuration from a YAML file."""
    with open(config_path, "r") as f:
        if config_path.endswith(".json"):
            config_dict = json.load(f)
        else:
            config_dict = yaml.safe_load(f)
    
    # Create config with default values
    config = JailbreakConfig()
    
    # Update with values from file
    for key, value in config_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Set up API keys from environment variables if not in config
    if config.api_keys is None:
        config.api_keys = {}
    
    for provider in ["OPENAI", "TOGETHER", "ANTHROPIC", "AZURE"]:
        env_key = f"{provider}_API_KEY"
        if env_key in os.environ and provider.lower() not in config.api_keys:
            config.api_keys[provider.lower()] = os.environ[env_key]
    
    return config


def save_config(config: JailbreakConfig, output_path: str) -> None:
    """Save configuration to a YAML file."""
    config_dict = {k: v for k, v in config.__dict__.items()}
    
    # Don't save API keys
    if "api_keys" in config_dict:
        config_dict["api_keys"] = {k: "***" for k in config_dict["api_keys"]}
    
    with open(output_path, "w") as f:
        if output_path.endswith(".json"):
            json.dump(config_dict, f, indent=2)
        else:
            yaml.dump(config_dict, f, default_flow_style=False) 