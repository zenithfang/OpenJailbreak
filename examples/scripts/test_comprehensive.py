#!/usr/bin/env python3
"""
Comprehensive Attack Testing Script (Attack-Agnostic)
Tests multiple models against multiple datasets with resumeable capability
Generates comprehensive Attack Success Rate (ASR) table

Usage:
    python examples/scripts/test_comprehensive.py --attack_name <attack> --dataset <dataset> [additional_args...]
    
Examples:
    python examples/scripts/test_comprehensive.py --attack_name ice --dataset harmbench --all_samples --eval_model gpt-4o
    python examples/scripts/test_comprehensive.py --attack_name many_shot --dataset jbb-harmful --samples 10 --verbose
    python examples/scripts/test_comprehensive.py --attack_name simple_override --all_samples --eval_provider wenwen
    
Features:
    - Tests multiple models × multiple datasets combinations
    - Attack-agnostic: supports any attack from universal_attack.py
    - Parallel execution for faster completion
    - Resumeable progress (saved to comprehensive_results.json)
    - Real-time ASR table updates
    - Markdown report generation (final_results.md)
    - 20-hour timeout per test
    - Automatic retry on network issues
    - Forwards arbitrary arguments to universal_attack.py
"""

import json
import os
import sys
import subprocess
import time
import argparse
import fcntl
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

class ComprehensiveAttackTester:
    """Attack-agnostic comprehensive testing framework"""
    
    # Hardcoded providers and datasets (configurable)
    PROVIDERS = [
        {
            "name": "gpt-3.5-turbo",
            "model": "gpt-3.5-turbo",
            "provider": "openai",
        },
        {
            "name": "gpt-4",
            "model": "gpt-4",
            "provider": "openai",
        },
        {
            "name": "gpt-4o",
            "model": "gpt-4o",
            "provider": "openai",
        },
    ]

    DATASETS = ["jbb-harmful", "advbench", "harmbench"]

    def __init__(self, attack_name, extra_args=None):
        self.attack_name = attack_name
        self.extra_args = extra_args or []
        self.base_output_dir = f"results/{attack_name}_comprehensive"
        self.results_file = os.path.join(self.base_output_dir, f"{attack_name}_comprehensive_results.json")
        
        # Ensure base directory exists
        os.makedirs(self.base_output_dir, exist_ok=True)

    def load_progress(self):
        """Load existing progress from JSON file with file locking"""
        try:
            with open(self.results_file, 'r+') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"completed": {}}

    def save_progress(self, progress):
        """Save progress to JSON file with atomic read-merge-write operation"""
        if not os.path.exists(self.results_file):
            with open(self.results_file, 'w') as f:
                json.dump({"completed": {}}, f, indent=2)
        
        with open(self.results_file, 'r+') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.seek(0)
            try:
                latest_progress = json.load(f)
            except (json.JSONDecodeError, ValueError):
                latest_progress = {"completed": {}}
            
            if "completed" not in latest_progress:
                latest_progress["completed"] = {}
            
            latest_progress["completed"].update(progress["completed"])
            
            f.seek(0)
            f.truncate()
            json.dump(latest_progress, f, indent=2)
            f.flush()

    def run_attack(self, provider_config, dataset, output_dir):
        """Run attack for a specific model-dataset combination"""
        print(f"🎯 Testing {self.attack_name} attack with {provider_config['name']} on {dataset}...")
        
        cmd = [
            "python", "examples/universal_attack.py",
            "--attack_name", self.attack_name,
            "--model", provider_config["model"],
            "--provider", provider_config["provider"],
            "--dataset", dataset,
            "--output_dir", output_dir,
            "--verbose"
        ]
        
        # Add provider-specific arguments
        for key in ["api_base", "api_key"]:
            if key in provider_config:
                cmd.extend([f"--{key}", provider_config[key]])
        
        # Add extra arguments
        cmd.extend(self.extra_args)
        
        try:
            print(f"🔥 Running command: {' '.join(cmd)}")
            print("-" * 80)
            result = subprocess.run(cmd, timeout=72000)
            return (True, "Success") if result.returncode == 0 else (False, f"Exit code: {result.returncode}")
        except subprocess.TimeoutExpired:
            return False, "Timeout after 20 hours"
        except Exception as e:
            return False, str(e)

    def extract_result(self, output_dir):
        """Extract ASR from results file by counting jailbroken results"""
        try:
            results_files = list(Path(output_dir).glob("results_*.json"))
            if not results_files:
                return {"asr": -1, "valid_count": 0, "invalid_count": 0, "prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            
            # Get most recent results file
            results_file = max(results_files, key=lambda f: f.stat().st_mtime)
            
            with open(results_file, 'r') as f:
                data = json.load(f)
                all_results = data.get("results", [])
                
                # Filter valid results and calculate metrics in one pass
                valid_results = []
                total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                
                for result in all_results:
                    if result.get("jailbroken") is not None:
                        valid_results.append(result)
                        usage = result.get("usage", {})
                        total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                        total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                        total_usage["total_tokens"] += usage.get("total_tokens", 0)
                
                valid_count = len(valid_results)
                invalid_count = len(all_results) - valid_count
                
                # Calculate ASR
                asr = sum(1 for r in valid_results if r.get("jailbroken", False)) / valid_count if valid_count > 0 else -1
                
                return {
                    "asr": asr,
                    "valid_count": valid_count,
                    "invalid_count": invalid_count,
                    "prompt_tokens": total_usage["prompt_tokens"],
                    "output_tokens": total_usage["completion_tokens"],
                    "total_tokens": total_usage["total_tokens"],
                }
                
        except Exception as e:
            print(f"Warning: Could not extract ASR from {output_dir}: {e}")
            return {"asr": -1, "valid_count": 0, "invalid_count": 0, "prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def print_progress_table(self, progress):
        """Print current progress as a table"""
        print(f"\n{'='*90}")
        print(f"{self.attack_name.upper()} Attack Progress - Attack Success Rate (ASR)")
        print("="*90)
        
        # Header
        print(f"{'Dataset':<15}", end="")
        for provider in self.PROVIDERS:
            print(f"{provider['name']:<14}", end="")
        print()
        
        print("-" * 90)
        
        # Rows
        for dataset in self.DATASETS:
            print(f"{dataset:<15}", end="")
            for provider in self.PROVIDERS:
                key = f"{provider['name']}_{dataset}"
                if key in progress["completed"]:
                    asr = progress["completed"][key]["asr"]
                    print(f"{asr:.1%}".ljust(14), end="")
                else:
                    print("PENDING".ljust(14), end="")
            print()
        
        print("="*90)

    def generate_markdown_report(self, progress):
        """Generate markdown report with progress table"""
        lines = [
            f"## {self.attack_name.upper()} Attack Success Rate (ASR) Results",
            "",
            "| Dataset | " + " | ".join(p["name"] for p in self.PROVIDERS) + " |",
            "|" + "---|" * (len(self.PROVIDERS) + 1)
        ]
        
        for dataset in self.DATASETS:
            row = f"| {dataset} |"
            for provider in self.PROVIDERS:
                key = f"{provider['name']}_{dataset}"
                if key in progress["completed"]:
                    asr = progress["completed"][key]["asr"]
                    valid_count = progress["completed"][key].get("valid_count", 0)
                    invalid_count = progress["completed"][key].get("invalid_count", 0)
                    if invalid_count > 0:
                        row += f" {asr:.1%} ({valid_count}v/{invalid_count}i) |"
                    else:
                        row += f" {asr:.1%} |"
                else:
                    row += " PENDING |"
            lines.append(row)
        
        return "\n".join(lines)

    def fix_progress_from_results(self):
        """Fix progress JSON by scanning existing results files"""
        print(f"🔧 Fixing progress JSON from existing results for {self.attack_name}...")
        
        progress = {"completed": {}}
        
        for provider_config in self.PROVIDERS:
            for dataset in self.DATASETS:
                combination_key = f"{provider_config['name']}_{dataset}"
                output_dir = os.path.join(self.base_output_dir, combination_key)
                
                if os.path.exists(output_dir):
                    result = self.extract_result(output_dir)
                    if result["asr"] > -1:
                        progress["completed"][combination_key] = {
                            "asr": result["asr"],
                            "valid_count": result["valid_count"],
                            "invalid_count": result["invalid_count"],
                            "prompt_tokens": result["prompt_tokens"],
                            "output_tokens": result["output_tokens"],
                            "total_tokens": result["total_tokens"],
                            "timestamp": datetime.now().isoformat(),
                            "model": provider_config["name"],
                            "provider": provider_config["provider"],
                            "dataset": dataset,
                            "attack_name": self.attack_name,
                            "fixed_from_results": True
                        }
                        print(f"✅ Found results for {combination_key}: ASR = {result['asr']:.1%}")
                    else:
                        print(f"⚠️  Found directory but no valid results for {combination_key}")
                else:
                    print(f"❌ No results found for {combination_key}")
        
        self.save_progress(progress)
        print(f"\n🎉 Updated {len(progress['completed'])} combinations in progress JSON")
        
        # Generate and save markdown report
        markdown_report = self.generate_markdown_report(progress)
        final_results_file = f"{self.base_output_dir}/final_results.md"
        with open(final_results_file, 'w') as f:
            f.write(markdown_report)
        print(f"📝 Markdown report saved to: {final_results_file}")
        
        self.print_progress_table(progress)

    def run_comprehensive_test(self, model_filter=None, dataset_filter=None):
        """Run comprehensive testing with optional filters"""
        providers = [p for p in self.PROVIDERS if not model_filter or p["name"] == model_filter]
        datasets = [d for d in self.DATASETS if not dataset_filter or d == dataset_filter]
        
        if not providers:
            print(f"❌ Model '{model_filter}' not found!")
            return
        if not datasets:
            print(f"❌ Dataset '{dataset_filter}' not found!")
            return
        
        print(f"🎯 {self.attack_name.upper()} Attack Comprehensive Testing")
        print("="*50)
        print(f"Testing {len(providers)} models × {len(datasets)} datasets")
        print(f"Total combinations: {len(providers) * len(datasets)}")
        print(f"Attack: {self.attack_name}")
        if model_filter:
            print(f"Model filter: {model_filter}")
        if dataset_filter:
            print(f"Dataset filter: {dataset_filter}")
        if self.extra_args:
            print(f"Extra arguments: {' '.join(self.extra_args)}")
        print()
        
        progress = self.load_progress()
        print(f"Loaded progress: {len(progress['completed'])} completed combinations")
        
        # Collect pending tasks
        tasks = []
        for provider_config in providers:
            for dataset in datasets:
                combination_key = f"{provider_config['name']}_{dataset}"
                
                if combination_key in progress["completed"]:
                    print(f"⏭️  Skipping {combination_key} (already completed)")
                    continue
                
                output_dir = os.path.join(self.base_output_dir, combination_key)
                os.makedirs(output_dir, exist_ok=True)
                tasks.append((provider_config, dataset, output_dir, combination_key))
        
        if not tasks:
            print("🎉 All combinations already completed!")
            self.print_progress_table(progress)
            return
        
        print(f"🚀 Starting {len(tasks)} tasks in parallel...")
        
        # Run all tasks in parallel
        total_combinations = len(providers) * len(datasets)
        completed_count = len(progress["completed"])
        
        with ProcessPoolExecutor(max_workers=len(tasks)) as executor:
            future_to_task = {
                executor.submit(self.run_attack, provider_config, dataset, output_dir): 
                (provider_config, dataset, output_dir, combination_key)
                for provider_config, dataset, output_dir, combination_key in tasks
            }
            
            for future in as_completed(future_to_task):
                provider_config, dataset, output_dir, combination_key = future_to_task[future]
                
                try:
                    success, output = future.result()
                    
                    if success:
                        result = self.extract_result(output_dir)
                        current_progress = self.load_progress()
                        current_progress["completed"][combination_key] = {
                            "asr": result["asr"],
                            "valid_count": result["valid_count"],
                            "invalid_count": result["invalid_count"],
                            "prompt_tokens": result["prompt_tokens"],
                            "output_tokens": result["output_tokens"],
                            "total_tokens": result["total_tokens"],
                            "timestamp": datetime.now().isoformat(),
                            "model": provider_config["name"],
                            "provider": provider_config["provider"],
                            "dataset": dataset,
                            "attack_name": self.attack_name,
                        }
                        print(f"✅ {combination_key}: ASR = {result['asr']:.1%}")
                        self.save_progress(current_progress)
                        completed_count += 1
                    else:
                        print(f"❌ {combination_key}: FAILED - {output[:100]}...")
                        print("⚠️  This test will be retried on next run.")
                    
                    print(f"Progress: {completed_count}/{total_combinations} ({completed_count/total_combinations:.1%})")
                    
                except Exception as e:
                    print(f"❌ {combination_key}: EXCEPTION - {str(e)}")
                    print("⚠️  This test will be retried on next run.")
        
        # Generate final report
        final_progress = self.load_progress()
        print(f"\n{'='*90}")
        print(f"🎉 {self.attack_name.upper()} ATTACK COMPREHENSIVE TESTING COMPLETED!")
        print("="*90)
        self.print_progress_table(final_progress)
        
        markdown_report = self.generate_markdown_report(final_progress)
        final_results_file = f"{self.base_output_dir}/final_results.md"
        with open(final_results_file, 'w') as f:
            f.write(markdown_report)
        
        print(f"\n📝 Markdown report saved to: {final_results_file}")
        print(f"Individual test results in: {self.base_output_dir}/")

def parse_args():
    """Parse command line arguments, separating known from unknown"""
    parser = argparse.ArgumentParser(
        description="Comprehensive Attack Testing Script (Attack-Agnostic)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--attack_name", required=True, help="Name of the attack to test")
    parser.add_argument("--dataset", help="Test only this dataset")
    parser.add_argument("--model", help="Test only this model")
    parser.add_argument("--status", action="store_true", help="Only print progress table and exit")
    parser.add_argument("--fix-progress", action="store_true", help="Fix progress JSON from existing results and exit")
    
    known_args, unknown_args = parser.parse_known_args()
    return known_args, unknown_args

def main():
    """Main function"""
    args, extra_args = parse_args()
    
    tester = ComprehensiveAttackTester(args.attack_name, extra_args)
    
    if args.status:
        progress = tester.load_progress()
        tester.print_progress_table(progress)
        return
    
    if args.fix_progress:
        tester.fix_progress_from_results()
        return
    
    tester.run_comprehensive_test(args.model, args.dataset)

if __name__ == "__main__":
    main() 