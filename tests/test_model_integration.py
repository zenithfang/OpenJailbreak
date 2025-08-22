#!/usr/bin/env python3
"""
Simple multi-provider test script for AutoJailbreak
Tests Aliyun, Azure, OpenAI, Bedrock, and Gemini providers
"""

import os
import sys
sys.path.append('src')

import autojailbreak as ajb

# Simple provider configurations
PROVIDERS = {
    # "llama2-7b-chat": {
    #     "model": "llama2-7b-chat",
    #     "api_base": "http://10.210.22.10:30254/v1",
    #     "provider": "openai",
    #     "prompt": "你是谁？",
    # },
    # "llama3-8b-instruct": {
    #     "model": "llama3-8b-instruct",
    #     "api_base": "http://10.210.22.10:30253/v1",
    #     "provider": "openai",
    #     "prompt": "你是谁？"
    # },
    "wenwen": {
        "model": "gpt-3.5-turbo",
        "api_key": os.getenv("WENWEN_API_KEY"),
        "api_base": "https://api.wenwen-ai.com/v1",
        "provider": "openai",
        "prompt": "你是谁？"
    },
    # "aliyun": {
    #     "model": "llama3.1-70b-instruct",
    #     "api_key": os.getenv("DASHSCOPE_API_KEY"),
    #     "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    #     "provider": "openai",
    #     "prompt": "你是谁？"
    # },
    # "openai": {
    #     "model": "gpt-4o",
    #     "provider": "openai",
    #     "prompt": "你是谁？"
    # },
    # "azure": {
    #     "model": "gpt-4o",
    #     "provider": "azure",
    #     "prompt": "你是谁？"
    # },
    # "bedrock": {
    #     "model": "us.anthropic.claude-sonnet-4-20250514-v1:0",
    #     "provider": "bedrock",
    #     "prompt": "你是谁？"
    # },
    # "claude3-7-sonnet": {
    #     "model": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    #     "provider": "bedrock",
    #     "prompt": "你是谁？"
    # },
    # "claude3-5-sonnet": {
    #     "model": "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
    #     "provider": "bedrock",
    #     "prompt": "你是谁？"
    # },
    # "gemini": {
    #     "model": "gemini-2.5-pro",  # litellm 格式，去掉 google/ 前缀
    #     "provider": "vertex_ai",
    #     "api_key": os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
    #     "prompt": "你是谁？"
    # },
    # "qwen3-8b": {
    #     "model": "qwen3-8b",
    #     "provider": "infini",
    #     "prompt": "bye"
    # },
    # "qwen3-14b": {
    #     "model": "qwen3-14b",
    #     "provider": "infini",
    #     "prompt": "如何杀人"
    # },
    # "deepseek-r1": {
    #     "model": "deepseek-r1-0528-qwen3-8b",
    #     "provider": "infini",
    #     "prompt": "你是谁？"
    # },
    "gpt-oss-120b": {
        "model": "gpt-oss-120b",
        "provider": "openai",
        "prompt": "你是谁？",
        "api_base": "http://10.210.22.10:30258/v1"
    }
}

def test_provider(name, config):
    """Test a single provider"""
    print(f"\n=== Testing {name.upper()} ===")
    
    try:
        # Build LLM parameters
        kwargs = {
            "model_name": config["model"],
            "provider": config["provider"],
            "log_dir": None
        }
        
        # Add API key if different from the default
        if "api_key" in config:
            kwargs["api_key"] = config["api_key"]
        
        # Add API base if needed
        if "api_base" in config:
            kwargs["api_base"] = config["api_base"]

        # Create LLM and test
        llm = ajb.LLMLiteLLM.from_config(**kwargs)
        response = llm.query(config["prompt"])
        

        # Check if response contains error indicators
        if isinstance(response, str):
            if response.startswith("Error:") or "Error" in response:
                print(f"❌ FAILED: {response}")
                return False
        print(f"✅ SUCCESS: {response}...")
        print(response.get_usage())
        print(response.get_reasoning_content())
        
        print(type(response))
        print('is str', isinstance(response, str))
        print('is dict', isinstance(response, dict))
        print('is list', isinstance(response, list))
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False

def main():
    print("🚀 AutoJailbreak Multi-Provider Test")
    print("=" * 50)
    
    results = {}
    for name, config in PROVIDERS.items():
        results[name] = test_provider(name, config)

    # Summary
    print("\n" + "=" * 50)
    print("📊 RESULTS:")
    passed = 0
    for name, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {name.upper()}")
        if success:
            passed += 1
    
    print(f"\n🎯 {passed}/{len(results)} tests passed")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)