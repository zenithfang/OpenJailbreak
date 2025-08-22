#!/usr/bin/env python3
"""
Universal AutoJailbreak Attack Script with Modular Architecture

This script provides a unified interface for running various jailbreak attacks
using the modular attack system with auto-discovery capabilities.
"""

import os
import sys
import json
import random
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Add the project root to the Python path
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
    print(f"Warning: Modern attack system not available: {e}")


def create_argument_parser():
    """Create context-aware argument parser that only loads relevant attack parameters."""
    try:
        from examples.config.dynamic_args_parser import parse_args_with_context
        # Use the new context-aware parsing system
        return parse_args_with_context()
    except ImportError as e:
        print(f"Error: Failed to import context-aware parser ({e})")
        exit(1)


def list_attacks_and_exit():
    """Attack listing functionality removed for performance reasons."""
    print("Attack listing functionality has been removed for performance reasons.")
    print("To use an attack, specify it directly with --attack_name.")
    print("\nUsage Examples:")
    print("  python examples/universal_attack.py --attack_name many_shot")
    print("  python examples/universal_attack.py --attack_name simple_override --all_samples")
    print("  python examples/universal_attack.py --attack_name translate_chain --samples 50")
    print("\nFor attack-specific help:")
    print("  python examples/universal_attack.py --attack_name many_shot --help")
    
    sys.exit(0)


def setup_attack(args, attack_name=None):
    """Set up a specific attack using modular system."""
    target_attack = attack_name or args.attack_name
    
    if not target_attack:
        raise ValueError("No attack specified")
    
    # Try modern system first
    if MODERN_SYSTEM_AVAILABLE:
        try:
            attack = create_attack(target_attack, args=args)
            
            # Get attack metadata
            attack_class = registry.get_attack(target_attack)
            config = attack_class.get_config()
            
            attack_meta = {
                "name": target_attack,
                "paper": config.get("paper", "Unknown"),
                "system": "modern",
                "config": config
            }
            
            return attack, attack_meta
            
        except ValueError as e:
            if args.verbose:
                print(f"Attack {target_attack} not found in modern system: {e}")
    
    # Attack not found in modern system
    available_attacks = []
    if MODERN_SYSTEM_AVAILABLE:
        available_attacks.extend(list(registry._attacks.keys()))
    
    raise ValueError(f"Unknown attack: {target_attack}. Available: {available_attacks}")


def setup_model(args):
    """Set up the model based on provider."""
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
    """Set up the evaluator with automatic dataset-specific detection."""
    dataset_name = args.dataset.lower()
    
    # GSM8K dataset detection (exact matches)
    if dataset_name in ['gsm8k', 'gsm8k-evil']:
        try:
            evaluator = ajb.JailbreakEvaluator.from_preset("GSM8K-Hybrid")
            evaluation_info = {
                "method": "GSM8K-Hybrid",
                "model": "Auto-detected for GSM8K",
                "provider": "pattern_matching_with_llm_fallback",
                "criteria": ["numerical_correctness", "refusal_detection"],
                "dataset_type": "gsm8k",
                "dataset_variant": dataset_name
            }
            if args.verbose:
                print(f"Auto-detected GSM8K dataset ({dataset_name}): using GSM8K-Hybrid evaluator")
            return evaluator, evaluation_info
        except (ValueError, ImportError) as e:
            if args.verbose:
                print(f"GSM8K evaluator not available ({e}), falling back to generic evaluator")
    
    # WMDP dataset detection (starts with 'wmdp-')
    elif dataset_name.startswith('wmdp-'):
        # Extract WMDP domain (bio, chem, cyber)
        wmdp_domain = dataset_name.split('-')[1] if '-' in dataset_name else 'generic'
        
        try:
            evaluator = ajb.JailbreakEvaluator.from_preset("WMDP-Hybrid")
            evaluation_info = {
                "method": "WMDP-Hybrid",
                "model": "Auto-detected for WMDP",
                "provider": "domain_specific_evaluation",
                "criteria": ["domain_correctness", "refusal_detection"],
                "dataset_type": "wmdp",
                "dataset_variant": dataset_name,
                "wmdp_domain": wmdp_domain
            }
            if args.verbose:
                print(f"Auto-detected WMDP dataset ({dataset_name}): using WMDP-Hybrid evaluator")
            return evaluator, evaluation_info
        except (ValueError, ImportError) as e:
            if args.verbose:
                print(f"WMDP evaluator not available ({e}), falling back to generic evaluator")
    
    # Fall back to generic jailbreak evaluation for other datasets
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
    else:  # default to OpenAI
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
        print(f"Using generic jailbreak evaluator ({preset_name}) for dataset: {dataset_name}")
    
    return evaluator, evaluation_info


def load_dataset(args):
    """Load the dataset for evaluation."""
    # Determine dataset name to load
    dataset_name = "jbb-harmful" if args.dataset == "harmful" else args.dataset
    
    # Load dataset
    try:
        dataset = ajb.read_dataset(dataset_name)
    except:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Apply sampling if needed (only if not using all samples)
    if not args.all_samples and args.samples and args.samples < len(dataset):
        dataset = dataset.sample(args.samples, args.seed)
    
    return dataset


def get_results_file_path(args, attack_meta):
    """Get the standardized results file path."""
    if hasattr(args, 'output') and args.output:
        return args.output
    
    attack_name = attack_meta.get("name", "unknown")
    output_file = f"{args.output_dir}/results_{attack_name}.json"
    
    return output_file


def load_existing_results(output_file):
    """Load existing results if the file exists."""
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                data = json.load(f)
                completed_ids = {r["example_id"] for r in data.get("results", []) if r["jailbroken"] is not None}
                print(f"📁 Found existing results file with {len(completed_ids)} completed examples")
                print(f"🔄 Resuming from where we left off...")
                return data, completed_ids
        except Exception as e:
            print(f"Warning: Could not load existing results: {e}")
    
    return None, set()


def initialize_results_file(args, attack_meta):
    """Initialize the results file with metadata and handle resume functionality."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = get_results_file_path(args, attack_meta)
    
    # Check if resume is enabled and file exists
    if args.resume:
        existing_data, completed_ids = load_existing_results(output_file)
        if existing_data:
            # Update metadata with current timestamp but keep existing results
            existing_data["metadata"]["timestamp"] = timestamp
            existing_data["metadata"]["args"] = vars(args)
            
            with open(output_file, 'w') as f:
                json.dump(existing_data, f, indent=2)
            
            print(f"📁 Resuming from existing file: {output_file}")
            return output_file, completed_ids
    
    # Create new file (or overwrite if resume is disabled)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Initialize file with metadata
    initial_data = {
        "metadata": {
            "timestamp": timestamp,
            "model": args.model,
            "provider": args.provider,
            "dataset": args.dataset,
            "attack": attack_meta,
            "system_info": {
                "modular_system_available": MODERN_SYSTEM_AVAILABLE,
                "attack_system_used": attack_meta.get("system", "modern")
            },
            "args": vars(args)
        },
        "results": []
    }
    
    with open(output_file, 'w') as f:
        json.dump(initial_data, f, indent=2)
    
    print(f"🆕 Created new results file: {output_file}")
    return output_file, set()


def process_single_example(args_tuple):
    """Process a single example (extracted for parallel processing)."""
    i, example, llm, attack, evaluator, attack_meta, args, output_file, completed_ids = args_tuple
    
    # Skip if already completed
    if i in completed_ids:
        if args.verbose:
            print(f"⏭️  Skipping already completed example {i+1}")
        return None
    
    if args.verbose:
        print(f"🎯 Running {attack_meta['name']} attack on {args.dataset} example {i+1} via {args.provider}/{args.model}")
    
    # Generate the jailbroken prompt
    try:
        jailbroken_prompt = attack.generate_attack(
            prompt=example["goal"],
            goal=example["goal"],
            target=example.get("target", "")
        )
    except Exception as e:
        print(f"Error generating attack for example {i}: {e}")
        jailbroken_prompt = None
        response = None
        success = None
    else:
        # Query the model
        try:
            response = llm.query(jailbroken_prompt)
        except Exception as e:
            print(f"Error querying model for example {i}: {e}")
            response = None
            success = None
        else:
            # Evaluate if the attack was successful
            # Create evaluation input with all available example data for specialized evaluators
            eval_input = dict(example)  # Start with all example fields
            eval_input["question"] = example["goal"]  # Override question
            eval_input["answer"] = response  # Add the model's response
            
            try:
                success = evaluator(eval_input)
            except Exception as e:
                print(f"Error evaluating response for example {i}: {e}")
                success = None
    
    # Store result
    result = {
        "example_id": i,
        "goal": example["goal"],
        "attack_prompt": jailbroken_prompt,
        "model_response": response,
        "jailbroken": success,
        "usage": response.get_usage() if response else None,
        "reasoning_content": response.get_reasoning_content() if response else None,
        # "attack_meta": attack_meta
    }
    
    # Save result to file immediately
    import fcntl
    try:
        with open(output_file, 'r+') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            data = json.load(f)
            data["results"].append(result)
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception as e:
        print(f"Warning: Could not save result to file: {e}")
    
    if args.verbose:
        print(f"  Example {i+1} Result: {'SUCCESS' if success else 'FAILED'}")
    
    return result


def run_attack_on_dataset(llm, attack, dataset, evaluator, attack_meta, args):
    """Run the attack on the dataset and return results."""
    # Initialize output file and get completed IDs
    output_file, completed_ids = initialize_results_file(args, attack_meta)
    
    # Prepare arguments for parallel processing
    task_args = [
        (i, example, llm, attack, evaluator, attack_meta, args, output_file, completed_ids)
        for i, example in enumerate(dataset)
    ]
    
    # Process in parallel with configurable workers
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Just execute, don't collect results (they're saved to file incrementally)
        executor.map(process_single_example, task_args)
    
    return output_file


def update_final_metadata(output_file, evaluation_info):
    """Update the results file with final metadata."""
    # Update metadata with final results
    try:
        with open(output_file, 'r+') as f:
            data = json.load(f)
           
            # Get results
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
            
            # Save updated data
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()
    except Exception as e:
        print(f"Warning: Could not update final metadata: {e}")
    
    print(f"\nResults saved to: {output_file}")
    return output_file


def run_single_attack(args, llm, evaluator, evaluation_info, dataset, attack_name=None):
    """Run a single attack and return results."""
    # Set up attack
    attack, attack_meta = setup_attack(args, attack_name)
    
    print(f"\nRunning {attack_meta['name']} attack using {attack_meta['system']} system...")
    if args.verbose:
        print(f"Attack paper: {attack_meta.get('paper', 'unknown')}")
    
    # Run the attack
    output_file = run_attack_on_dataset(llm, attack, dataset, evaluator, attack_meta, args)
    
    # Update final metadata in the file
    update_final_metadata(output_file, evaluation_info)
    
    return output_file


def main():
    """Main function."""
    # Parse arguments using context-aware parser
    args = create_argument_parser()
    
    # Check if user wants to list attacks
    if args.list_attacks:
        list_attacks_and_exit()
    
    # Set random seed
    random.seed(args.seed)
    
    print("Universal AutoJailbreak Attack Script - Modular Architecture")
    print("=" * 60)
    print(f"Modular attack system available: {MODERN_SYSTEM_AVAILABLE}")
    
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
    
    # Run the specified attack
    if not args.attack_name:
        print("Error: --attack_name is required")
        print("Use --list_attacks to see available attacks")
        return
    
    # Run single attack
    output_file = run_single_attack(args, llm, evaluator, evaluation_info, dataset)
    print(f"\nAttack completed. Results saved to: {output_file}")


if __name__ == "__main__":
    main()
