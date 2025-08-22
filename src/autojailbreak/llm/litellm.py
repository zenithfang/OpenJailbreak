"""
LLM interface for API-based models using LiteLLM.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Union, Any
import time

# Import litellm with a try-except to provide informative error
try:
    import litellm
    from litellm import completion
    litellm.suppress_debug_info = True
    litellm.drop_params = True
except ImportError:
    raise ImportError(
        "LiteLLM is required to use LLMLiteLLM. "
        "Install it with `pip install litellm`."
    )

from ..llm.base import BaseLLM
from ..defenses import get_defense

# Default defense configurations (moved from config.py)
DEFAULT_DEFENSE_CONFIGS = {
    "smoothllm": {
        "n_samples": 10,
        "sigma": 0.5,
        "p": 0.3
    },
    "perplexity": {
        "threshold": 10.0,
        "sliding_window": 5
    },
    "retokenization": {
        "n_retokenizations": 3
    },
    "dictionary": {
        "min_word_length": 3,
        "non_dict_threshold": 0.1
    }
}

class RichResponse(str):
    """
    A string subclass that contains both the response text and usage information.
    This class behaves exactly like a string for backward compatibility.
    """
    def __new__(cls, text: str, usage: Dict[str, int], reasoning_content: str):
        # Create a new string instance
        instance = super().__new__(cls, text)
        instance.usage = usage
        instance.reasoning_content = reasoning_content
        return instance
    
    def get_usage(self) -> Dict[str, int]:
        """Get the token usage information."""
        return self.usage.copy()
    
    def get_reasoning_content(self) -> str:
        """Get the reasoning content."""
        return self.reasoning_content

class LLMLiteLLM(BaseLLM):
    """
    Implementation of BaseLLM using LiteLLM for API-based models.
    
    This class supports various model providers like OpenAI, Together, Anthropic, etc.
    """
    
    @classmethod
    def from_config(
        cls,
        model_name: str,
        provider: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs
    ) -> "LLMLiteLLM":
        """
        Create an LLMLiteLLM instance from a user-friendly configuration.

        This factory method handles provider-specific logic, such as API key
        environment variables and API base URL overrides for certain providers.

        Args:
            model_name: Name of the model.
            provider: User-facing provider name (e.g., 'openai', 'aliyun').
            api_key: Optional API key.
            api_base: Optional API base URL.
            **kwargs: Additional parameters to pass to the LLM.

        Returns:
            An instance of LLMLiteLLM.
        """
        llm_provider = provider
        llm_api_key = api_key
        llm_api_base = api_base
        
        init_kwargs = kwargs.copy()

        if provider == "openai":
            llm_api_key = api_key or os.getenv("OPENAI_API_KEY")
            llm_api_base = api_base or os.getenv("OPENAI_API_BASE")
        elif provider == "anthropic":
            llm_api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        elif provider == "azure":
            llm_api_key = api_key or os.getenv("AZURE_API_KEY")
            llm_api_base = api_base or os.getenv("AZURE_API_BASE")
        elif provider == "aliyun":
            llm_provider = "openai"
            llm_api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
            llm_api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        elif provider == "wenwen":
            llm_provider = "openai"
            llm_api_key = api_key or os.getenv("WENWEN_API_KEY")
            llm_api_base = "https://api.wenwen-ai.com/v1"
        elif provider == "infini":
            llm_provider = "openai"
            llm_api_key = api_key or os.getenv("INFINI_API_KEY")
            llm_api_base = "https://cloud.infini-ai.com/maas/v1"
        elif provider == "bedrock":
            llm_api_key = api_key or os.getenv("AWS_ACCESS_KEY_ID")
        elif provider == "vertex_ai":
            llm_api_key = api_key or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        
        return cls(
            model_name=model_name,
            provider=llm_provider,
            api_key=llm_api_key,
            api_base=llm_api_base,
            **init_kwargs
        )

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        provider: Optional[str] = None,
        temperature: float = 1,
        max_tokens: Optional[int] = None,
        max_retries: int = 3,
        log_dir: Optional[str] = "logs",
        **kwargs
    ):
        """
        Initialize the LiteLLM-based LLM.
        
        Args:
            model_name: Name of the model to use
            api_key: API key for the provider
            api_base: Base URL for the API
            provider: Provider of the model
            temperature: Temperature parameter for generation
            max_tokens: Maximum number of tokens to generate
            max_retries: Maximum number of retries on API failures (default: 3)
            log_dir: Directory to store logs
            **kwargs: Additional parameters to pass to LiteLLM
        """
        super().__init__(model_name, **kwargs)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.log_dir = log_dir
        
        if provider:
            self.model_name = f"{provider}/{model_name}"
        else:
            raise ValueError(f"Provider not provided.")
        
        # Set up API key and base url if provided
        self.api_key = api_key
        self.api_base = api_base
        
        # Set up logging
        if log_dir:
            os.makedirs(f'{log_dir}/{provider}', exist_ok=True)
        
        # Store additional parameters
        self.kwargs = kwargs
        
        # Initialize query counter
        self.query_count = 0
    
    def _make_messages(self, prompt: str) -> List[Dict[str, str]]:
        """
        Format a prompt as a list of messages for chat-based models.
        
        Args:
            prompt: Raw prompt text
            
        Returns:
            List of message dictionaries
        """
        # Check if prompt already contains role prefixes
        if "SYSTEM:" in prompt or "USER:" in prompt or "ASSISTANT:" in prompt:
            # Parse from role prefixes
            messages = []
            lines = prompt.split("\n")
            current_role = None
            current_content = []
            
            for line in lines:
                if line.startswith("SYSTEM:"):
                    if current_role:
                        messages.append({"role": current_role, "content": "\n".join(current_content).strip()})
                    current_role = "system"
                    current_content = [line[7:].strip()]
                elif line.startswith("USER:"):
                    if current_role:
                        messages.append({"role": current_role, "content": "\n".join(current_content).strip()})
                    current_role = "user"
                    current_content = [line[5:].strip()]
                elif line.startswith("ASSISTANT:"):
                    if current_role:
                        messages.append({"role": current_role, "content": "\n".join(current_content).strip()})
                    current_role = "assistant"
                    current_content = [line[10:].strip()]
                elif current_role:
                    current_content.append(line)
            
            if current_role:
                messages.append({"role": current_role, "content": "\n".join(current_content).strip()})
            
            # If no system message, add a default one
            if not any(m["role"] == "system" for m in messages):
                messages.insert(0, {"role": "system", "content": "You are a helpful assistant."})
            
            return messages
        
        # Simple format: user message only
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    
    def query(
        self,
        prompts: Union[str, List[str]],
        behavior: Optional[str] = None,
        defense: Optional[str] = None,
        phase: Optional[str] = None,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Query the model with a prompt or list of prompts.
        
        Args:
            prompts: Prompt(s) to send to the model
            behavior: Behavior context for logging and evaluation
            defense: Defense to apply (if any)
            phase: Phase of querying (e.g., "test", "optimization")
            **kwargs: Additional query parameters
            
        Returns:
            Model response(s)
        """
        # Handle single prompt
        if isinstance(prompts, str):
            single_input = True
            prompts_list = [prompts]
        else:
            single_input = False
            prompts_list = prompts
        
        # Apply defense if specified
        if defense:
            defense_obj = get_defense(
                defense, 
                **DEFAULT_DEFENSE_CONFIGS.get(defense, {})
            )
            prompts_list = [defense_obj.apply(p) for p in prompts_list]
        
        responses = []
        for i, prompt in enumerate(prompts_list):
            try:
                # Format prompt as messages
                messages = self._make_messages(prompt)
                
                # Query the model
                self.query_count += 1
                query_id = f"{self.model_name}_{self.query_count}_{int(time.time())}"
                
                # Log the query
                if self.log_dir:
                    log_path = os.path.join(self.log_dir, f"{query_id}_query.json")
                    with open(log_path, "w") as f:
                        json.dump({
                            "model": self.model_name,
                            "messages": messages,
                            "behavior": behavior,
                            "defense": defense,
                            "phase": phase,
                        }, f, indent=2)
                
                # Make the API call
                completion_kwargs = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": kwargs.get("temperature", self.temperature),
                    "api_key": self.api_key or "sk-dummy-key-for-local-endpoint",
                    "api_base": self.api_base,
                }
                
                # Add optional parameters only if provided
                if max_tokens := kwargs.get("max_tokens", self.max_tokens):
                    completion_kwargs["max_tokens"] = max_tokens
                    
                # Add any additional kwargs passed during initialization
                for key, value in self.kwargs.items():
                    if key not in completion_kwargs:
                        completion_kwargs[key] = value
                    
                # print(completion_kwargs)
                
                # Retry logic for API calls
                last_error = None
                for attempt in range(self.max_retries + 1):
                    try:
                        response = completion(
                            **completion_kwargs
                        )
                        # print(completion_kwargs)
                        # print(response.usage)
                        # print(response.choices[0].message.reasoning_content)
                        break  # Success, exit retry loop
                    except Exception as e:
                        last_error = e
                        if attempt < self.max_retries:
                            wait_time = 2 ** attempt  # Exponential backoff
                            logging.warning(f"API call failed (attempt {attempt + 1}/{self.max_retries + 1}): {str(e)}. Retrying in {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            # Final attempt failed, re-raise the last error
                            raise last_error
                
                # Extract the response text
                response_text = response.choices[0].message.content
                reasoning_content = response.choices[0].message.reasoning_content if hasattr(response.choices[0].message, "reasoning_content") else ""
                
                # Apply defense processing if applicable
                if defense:
                    response_text = defense_obj.process_response(response_text)
                
                # Log the response
                if self.log_dir:
                    log_path = os.path.join(self.log_dir, f"{query_id}_response.json")
                    with open(log_path, "w") as f:
                        json.dump({
                            "model": self.model_name,
                            "response": response_text,
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens,
                        }, f, indent=2)
                
                # Create usage dictionary
                usage_dict = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
                responses.append(RichResponse(response_text, usage_dict, reasoning_content))
                
            except Exception as e:
                logging.error(f"Error querying model: {str(e)}, completion_kwargs: {json.dumps(completion_kwargs, indent=2)}")
                error_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                responses.append(RichResponse(f"Error: {str(e)}", error_usage, ""))
        
        # Return results
        if single_input:
            return responses[0]
        return responses
    
    def get_token_count(self, text: str) -> int:
        """
        Count the number of tokens in the text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        try:
            return litellm.utils.token_counter(model=self.model_name, text=text)
        except Exception as e:
            logging.warning(f"Error counting tokens: {str(e)}")
            # Fallback to approximate token count
            return len(text.split()) * 1.33  # Rough approximation 