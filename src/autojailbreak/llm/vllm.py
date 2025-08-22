"""
Dummy implementation of the vLLM interface for local models.
This is a placeholder to satisfy imports when local model inference is not needed.
"""

from typing import Dict, List, Optional, Union, Any
from ..llm.base import BaseLLM

class LLMvLLM(BaseLLM):
    """
    Dummy implementation of BaseLLM using vLLM for local models.
    Raises NotImplementedError when methods are called.
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the dummy vLLM-based LLM.
        
        Args:
            model_name: Name of the model to use
            **kwargs: Additional parameters
        """
        super().__init__(model_name, **kwargs)
        raise NotImplementedError(
            "vLLM is not available. For local inference, please install vLLM or use a remote provider (azure/openai/together)."
        )
    
    def query(self, prompts: Union[str, List[str]], **kwargs) -> Union[str, List[str]]:
        """
        Query the model with a prompt or list of prompts.
        
        Args:
            prompts: Prompt(s) to send to the model
            **kwargs: Additional query parameters
            
        Returns:
            Model response(s)
        """
        raise NotImplementedError("vLLM is not available")
    
    def get_token_count(self, text: str) -> int:
        """
        Count the number of tokens in the text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        raise NotImplementedError("vLLM is not available") 