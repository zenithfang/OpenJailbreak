#!/usr/bin/env python3
"""
Series attack script - a general framework for composing multiple attack methods.

This script provides a unified interface to run series attacks, where multiple attack methods are applied sequentially.
The output of the first attack becomes the input to the second, and so on. Finally, the output of the last attack is used for evaluation.
"""

import os
import sys
import json
import random
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.autojailbreak as ajb

# Try to import the modern attack system
try:
    from src.autojailbreak.attacks.registry import registry
    from src.autojailbreak.attacks.factory import AttackFactory
    create_attack = AttackFactory.create_attack
    MODERN_SYSTEM_AVAILABLE = True
except ImportError as e:
    MODERN_SYSTEM_AVAILABLE = False
    print(f"Warning: modern attack system unavailable: {e}")


def create_argument_parser():
    """Create an argument parser that supports series-attack specific options."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Series attack script - compose multiple attack methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples/universal_attack_series.py --config_file examples/configs/series_simple.json
  python examples/universal_attack_series.py --attack_chain "simple_override,translate_attack" --dataset harmful --samples 5
  python examples/universal_attack_series.py --attack_chain "many_shot,deepinception" --model gpt-3.5-turbo --provider wenwen
        """
    )
    
    # Basic configuration
    parser.add_argument("--config_file", type=str, help="Series attack config file path (JSON)")
    parser.add_argument("--attack_chain", type=str, help="Attack chain, comma-separated (e.g., 'simple_override,translate_attack')")
    
    # Model configuration
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="Target model name")
    parser.add_argument("--provider", type=str, default="wenwen", help="Model provider")
    parser.add_argument("--api_key", type=str, help="API key")
    parser.add_argument("--api_base", type=str, help="API base URL")
    
    # Dataset configuration
    parser.add_argument("--dataset", type=str, default="harmful", help="Dataset name")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples")
    parser.add_argument("--all_samples", action="store_true", help="Use all samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Evaluation configuration
    parser.add_argument("--eval_provider", type=str, default="openai", help="Evaluation provider")
    parser.add_argument("--eval_model", type=str, default="gpt-4o", help="Evaluation model name")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--resume", action="store_true", help="Resume from existing results file")
    
    # Execution configuration
    parser.add_argument("--max_workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--dry_run", action="store_true", help="Dry run; do not execute attacks")
    
    # Error handling
    parser.add_argument("--continue_on_error", action="store_true", help="Continue when an individual attack fails")
    parser.add_argument("--max_retry_attempts", type=int, default=2, help="Maximum retry attempts")
    
    return parser.parse_args()


def load_series_config(config_file: str) -> Dict[str, Any]:
    """Load the series attack configuration file."""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Validate configuration file
    required_fields = ["attack_chain"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Configuration file missing required field: {field}")
    
    if not config["attack_chain"]:
        raise ValueError("Attack chain cannot be empty")
    
    return config


def parse_attack_chain(attack_chain_str: str) -> List[str]:
    """Parse an attack chain string."""
    attacks = [attack.strip() for attack in attack_chain_str.split(',')]
    attacks = [attack for attack in attacks if attack]  # Remove empty strings
    
    if not attacks:
        raise ValueError("Attack chain cannot be empty")
    
    return attacks


def setup_attack_from_config(attack_name: str, attack_config: Dict[str, Any], args) -> Tuple[Any, Dict[str, Any]]:
    """Set up a single attack from the configuration."""
    if not MODERN_SYSTEM_AVAILABLE:
        raise ImportError("Modern attack system unavailable")
    
    try:
        # Merge attack-specific configuration into args
        class ConfigArgs:
            def __init__(self, base_args, attack_config):
                # Copy base arguments
                for attr in dir(base_args):
                    if not attr.startswith('_'):
                        setattr(self, attr, getattr(base_args, attr))
                
                # Apply attack-specific configuration
                for key, value in attack_config.items():
                    setattr(self, key, value)
        
        attack_args = ConfigArgs(args, attack_config)
        attack = create_attack(attack_name, args=attack_args)
        
        # Retrieve attack metadata
        attack_class = registry.get_attack(attack_name)
        config = attack_class.get_config()
        
        attack_meta = {
            "name": attack_name,
            "paper": config.get("paper", "Unknown"),
            "system": "modern",
            "config": attack_config
        }
        
        return attack, attack_meta
        
    except ValueError as e:
        available_attacks = list(registry._loaded_attacks.keys()) if MODERN_SYSTEM_AVAILABLE else []
        raise ValueError(f"Unknown attack: {attack_name}. Available attacks: {available_attacks}") from e


def setup_model(args):
    """Set up the model."""
    if args.provider == "local":
        return ajb.LLMvLLM(model_name=args.model)
    
    try:
        return ajb.LLMLiteLLM.from_config(
            model_name=args.model,
            provider=args.provider,
            api_key=args.api_key,
            api_base=args.api_base,
        )
    except Exception as e:
        raise ValueError(f"Failed to set up model for provider {args.provider}: {e}")


def setup_evaluator(args):
    """Set up the evaluator."""
    dataset_name = args.dataset.lower()
    
    # GSM8K dataset detection
    if dataset_name in ['gsm8k', 'gsm8k-evil']:
        try:
            evaluator = ajb.JailbreakEvaluator.from_preset("GSM8K-Hybrid")
            evaluation_info = {
                "method": "GSM8K-Hybrid",
                "model": "GSM8K auto-detection",
                "provider": "pattern_matching_with_llm_fallback",
                "criteria": ["numerical_correctness", "refusal_detection"],
                "dataset_type": "gsm8k",
                "dataset_variant": dataset_name
            }
            if args.verbose:
                print(f"Automatically detected GSM8K dataset ({dataset_name}): using GSM8K-Hybrid evaluator")
            return evaluator, evaluation_info
        except (ValueError, ImportError) as e:
            if args.verbose:
                print(f"GSM8K evaluator unavailable ({e}), falling back to generic evaluator")
    
    # WMDP dataset detection
    elif dataset_name.startswith('wmdp-'):
        wmdp_domain = dataset_name.split('-')[1] if '-' in dataset_name else 'generic'
        
        try:
            evaluator = ajb.JailbreakEvaluator.from_preset("WMDP-Hybrid")
            evaluation_info = {
                "method": "WMDP-Hybrid",
                "model": "WMDP auto-detection",
                "provider": "domain_specific_evaluation",
                "criteria": ["domain_correctness", "refusal_detection"],
                "dataset_type": "wmdp",
                "dataset_variant": dataset_name,
                "wmdp_domain": wmdp_domain
            }
            if args.verbose:
                print(f"Automatically detected WMDP dataset ({dataset_name}): using WMDP-Hybrid evaluator")
            return evaluator, evaluation_info
        except (ValueError, ImportError) as e:
            if args.verbose:
                print(f"WMDP evaluator unavailable ({e}), falling back to generic evaluator")
    
    # Fallback to generic jailbreak evaluation
    eval_provider = getattr(args, 'eval_provider', 'openai')
    
    if eval_provider == "azure":
        evaluator = ajb.JailbreakEvaluator.from_preset("AzureOpenAIChat-recheck")
        preset_name = "AzureOpenAIChat-recheck"
    elif eval_provider == "anthropic":
        evaluator = ajb.JailbreakEvaluator.from_preset("AnthropicChat-recheck")
        preset_name = "AnthropicChat-recheck"
    elif eval_provider == "wenwen":
        evaluator = ajb.JailbreakEvaluator.from_preset("WenwenOpenAIChat-recheck")
        preset_name = "WenwenOpenAIChat-recheck"
    else:  # Default OpenAI
        evaluator = ajb.JailbreakEvaluator.from_preset("OpenAIChat-recheck")
        preset_name = "OpenAIChat-recheck"
    
    evaluation_info = {
        "method": preset_name,
        "model": getattr(args, 'eval_model', 'gpt-4'),
        "provider": eval_provider,
        "criteria": ["harmfulness", "instruction_following"],
        "dataset_type": "generic"
    }
    
    if args.verbose:
        print(f"Using generic jailbreak evaluator for dataset ({preset_name}): {dataset_name}")
    
    return evaluator, evaluation_info


def load_dataset(args):
    """Load the dataset."""
    dataset_name = "jbb-harmful" if args.dataset == "harmful" else args.dataset
    
    try:
        dataset = ajb.read_dataset(dataset_name)
    except:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Apply sampling
    if not args.all_samples and args.samples and args.samples < len(dataset):
        dataset = dataset.sample(args.samples, args.seed)
    
    return dataset


def execute_attack_series(attacks_config: List[Dict[str, Any]], example: Dict[str, Any], 
                         llm, args) -> Tuple[str, List[Dict[str, Any]]]:
    """Execute the series attack chain."""
    current_prompt = example["goal"]
    goal = example["goal"] 
    target = example.get("target", "")
    attack_results = []
    
    if args.verbose:
        print(f"  Starting series attack chain with {len(attacks_config)} attacks")
        print(f"  Initial prompt: {current_prompt[:100]}...")
    
    for i, attack_config in enumerate(attacks_config):
        attack_name = attack_config["name"]
        attack_params = attack_config.get("parameters", {})
        
        if args.verbose:
            print(f"    Step {i+1}/{len(attacks_config)}: executing {attack_name} attack")
        
        if args.dry_run:
            # Dry-run mode
            result = {
                "step": i + 1,
                "attack_name": attack_name,
                "input_prompt": current_prompt,
                "output_prompt": f"[Dry run] Output of {attack_name}",
                "parameters": attack_params,
                "success": True,
                "error": None
            }
            current_prompt = result["output_prompt"]
            attack_results.append(result)
            continue
        
        # Set up attack
        retry_count = 0
        max_retries = args.max_retry_attempts
        
        while retry_count <= max_retries:
            try:
                attack, attack_meta = setup_attack_from_config(attack_name, attack_params, args)
                
                # Execute attack
                output_prompt = attack.generate_attack(
                    prompt=current_prompt,
                    goal=goal,
                    target=target
                )
                
                result = {
                    "step": i + 1,
                    "attack_name": attack_name,
                    "input_prompt": current_prompt,
                    "output_prompt": output_prompt,
                    "parameters": attack_params,
                    "success": True,
                    "error": None,
                    "meta": attack_meta
                }
                
                # Handle both string and list outputs from attacks
                if isinstance(output_prompt, list):
                    # For list outputs, use the first item or join them
                    current_prompt = output_prompt[0] if output_prompt else current_prompt
                else:
                    current_prompt = output_prompt
                attack_results.append(result)
                
                if args.verbose:
                    print(f"      ✅ {attack_name} succeeded")
                    if isinstance(output_prompt, list):
                        print(f"      Output length: {len(output_prompt)} samples")
                    else:
                        print(f"      Output length: {len(output_prompt)} characters")
                
                break  # Successful, exit retry loop
                
            except Exception as e:
                retry_count += 1
                error_msg = str(e)
                
                if retry_count <= max_retries:
                    if args.verbose:
                        print(f"      ⚠️ {attack_name} failed (attempt {retry_count}/{max_retries + 1}): {error_msg}")
                    continue
                else:
                    # Reached maximum retry attempts
                    if args.verbose:
                        print(f"      ❌ {attack_name} failed after retries: {error_msg}")
                    
                    result = {
                        "step": i + 1,
                        "attack_name": attack_name,
                        "input_prompt": current_prompt,
                        "output_prompt": None,
                        "parameters": attack_params,
                        "success": False,
                        "error": error_msg
                    }
                    attack_results.append(result)
                    
                    if not args.continue_on_error:
                        raise Exception(f"Attack {attack_name} failed; stopping chain: {error_msg}")
                    else:
                        # Continue using current prompt
                        if args.verbose:
                            print(f"      ⏭️ Continuing to next attack (using current prompt)")
                        break
    
    return current_prompt, attack_results


def process_single_example(args_tuple):
    """Process a single example (for parallel processing)."""
    (i, example, attacks_config, llm, evaluator, args, output_file, completed_ids) = args_tuple
    
    # Skip completed examples
    if i in completed_ids:
        if args.verbose:
            print(f"⏭️  Skipping completed example {i+1}")
        return None
    
    if args.verbose:
        print(f"🎯 Running series attack on example {i+1}")
        print(f"   Goal: {example['goal'][:100]}...")
    
    try:
        # Execute series attack
        final_prompt, attack_results = execute_attack_series(
            attacks_config, example, llm, args
        )
        
        if final_prompt is None or not final_prompt.strip():
            # Series attack chain failed
            response = None
            success = None
            if args.verbose:
                print(f"  Example {i+1} chain failed")
        else:
            # Query the model
            try:
                response = llm.query(final_prompt)
            except Exception as e:
                print(f"Model query failed for example {i}: {e}")
                response = None
                success = None
            else:
                # Evaluate whether the attack succeeded
                eval_input = dict(example)
                eval_input["question"] = example["goal"]
                eval_input["answer"] = response
                
                try:
                    success = evaluator(eval_input)
                except Exception as e:
                    print(f"Evaluation failed for example {i}: {e}")
                    success = None
    
    except Exception as e:
        print(f"Error processing example {i}: {e}")
        final_prompt = None
        response = None
        success = None
        attack_results = []
    
    # Store results
    result = {
        "example_id": i,
        "goal": example["goal"],
        "attack_chain": [r["attack_name"] for r in attack_results],
        "attack_details": attack_results,
        "final_attack_prompt": final_prompt,
        "model_response": response,
        "jailbroken": success,
        "usage": response.get_usage() if response else None,
        "reasoning_content": response.get_reasoning_content() if response else None,
    }
    
    # Immediately save results to file
    import fcntl
    try:
        with open(output_file, 'r+') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            data = json.load(f)
            data["results"].append(result)
            f.seek(0)
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.truncate()
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception as e:
        print(f"Warning: Failed to save results to file: {e}")
    
    if args.verbose:
        success_status = 'Success' if success else 'Failure' if success is not None else 'Error'
        print(f"  Example {i+1} result: {success_status}")
    
    return result


def initialize_results_file(args, attacks_config: List[Dict[str, Any]]):
    """Initialize the results file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if hasattr(args, 'output') and args.output:
        output_file = args.output
    else:
        attack_names = [config["name"] for config in attacks_config]
        chain_name = "_".join(attack_names)
        output_file = f"{args.output_dir}/results_series_{chain_name}.json"
    
    # Check if resume is enabled and the file exists
    if args.resume and os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                existing_data = json.load(f)
                completed_ids = {r["example_id"] for r in existing_data.get("results", []) if r["jailbroken"] is not None}
                print(f"📁 Found existing results file with {len(completed_ids)} completed examples")
                print(f"🔄 Resuming from checkpoint...")
                
                # Update timestamp
                existing_data["metadata"]["timestamp"] = timestamp
                existing_data["metadata"]["args"] = vars(args)
                
                with open(output_file, 'w') as f:
                    json.dump(existing_data, f, indent=2, ensure_ascii=False)
                
                print(f"📁 Resumed from existing file: {output_file}")
                return output_file, completed_ids
        except Exception as e:
            print(f"Warning: Failed to load existing results: {e}")
    
    # Create new file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    initial_data = {
        "metadata": {
            "timestamp": timestamp,
            "model": args.model,
            "provider": args.provider,
            "dataset": args.dataset,
            "attack_chain": attacks_config,
            "system_info": {
                "modular_system_available": MODERN_SYSTEM_AVAILABLE,
                "series_attack": True
            },
            "args": vars(args)
        },
        "results": []
    }
    
    with open(output_file, 'w') as f:
        json.dump(initial_data, f, indent=2, ensure_ascii=False)
    
    print(f"🆕 Created new results file: {output_file}")
    return output_file, set()


def update_final_metadata(output_file, evaluation_info):
    """Update the final metadata of the results file."""
    try:
        with open(output_file, 'r+') as f:
            data = json.load(f)
           
            results = data["results"]
            
            # Calculate success rate
            valid_count = sum(1 for r in results if r["jailbroken"] is not None)
            invalid_count = len(results) - valid_count
            success_count = sum(1 for r in results if r["jailbroken"])
            success_rate = success_count / valid_count if valid_count > 0 else 0
            
            print(f"Attack success rate: {success_rate:.2%} ({success_count}/{valid_count})")
            
            # Update metadata
            data["metadata"]["valid_count"] = valid_count
            data["metadata"]["invalid_count"] = invalid_count
            data["metadata"]["success_rate"] = success_rate
            data["metadata"]["evaluation"] = evaluation_info
            
            # Sort results by example_id
            data["results"].sort(key=lambda x: x["example_id"])
            
            # Save the updated data
            f.seek(0)
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.truncate()
    except Exception as e:
        print(f"Warning: Failed to update final metadata: {e}")
    
    print(f"\nResults saved to: {output_file}")
    return output_file


def run_series_attack_on_dataset(llm, attacks_config, dataset, evaluator, args):
    """Run the series attack on the dataset."""
    # Initialize the output file
    output_file, completed_ids = initialize_results_file(args, attacks_config)
    
    # Prepare parameters for parallel processing
    task_args = [
        (i, example, attacks_config, llm, evaluator, args, output_file, completed_ids)
        for i, example in enumerate(dataset)
    ]
    
    # Parallel processing
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        executor.map(process_single_example, task_args)
    
    return output_file


def main():
    """Main function."""
    args = create_argument_parser()
    
    # Set random seed
    random.seed(args.seed)
    
    print("Series attack script - compose multiple attack methods")
    print("=" * 60)
    print(f"Modern attack system available: {MODERN_SYSTEM_AVAILABLE}")
    
    # Configure the series attack
    if args.config_file:
        print(f"Loading from config file: {args.config_file}")
        config = load_series_config(args.config_file)
        attacks_config = config["attack_chain"]
    elif args.attack_chain:
        print(f"Loading attack chain from CLI: {args.attack_chain}")
        attack_names = parse_attack_chain(args.attack_chain)
        attacks_config = [{"name": name, "parameters": {}} for name in attack_names]
    else:
        print("Error: must specify --config_file or --attack_chain")
        return
    
    print(f"Attack chain contains {len(attacks_config)} attacks:")
    for i, attack_config in enumerate(attacks_config):
        print(f"  {i+1}. {attack_config['name']}")
    
    if args.dry_run:
        print("\n⚠️ Dry run - no attacks will be executed")
    
    # Set up model
    print("\nSetting up model...")
    llm = setup_model(args)
    
    # Set up evaluator
    print("Setting up evaluator...")
    evaluator, evaluation_info = setup_evaluator(args)
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(args)
    if args.all_samples:
        print(f"Loaded all {len(dataset)} examples from {args.dataset}")
    else:
        print(f"Loaded {len(dataset)} examples from {args.dataset} (limited by --samples={args.samples})")
    
    # Run series attack
    print(f"\nRunning series attack...")
    output_file = run_series_attack_on_dataset(llm, attacks_config, dataset, evaluator, args)
    
    # Update final metadata
    update_final_metadata(output_file, evaluation_info)
    print(f"\nSeries attack completed. Results saved to: {output_file}")


if __name__ == "__main__":
    main()
