#!/usr/bin/env python3
"""
Simple script to run all model combinations for level 1 recipes.
Runs each recipe 2x with all possible model combinations.
Based on the user's command template and existing scripts.
"""

import subprocess
import sys
import os
from datetime import datetime
from itertools import product

# Level 1 recipes (from convert_result.py)
LEVEL_1_RECIPES = [
    "baked_bell_pepper",
    "baked_sweet_potato", 
    "boiled_egg",
    "boiled_mushroom",
    "boiled_sweet_potato",
]

# Available models (based on your command using qwen/qwen3-32b and the existing scripts)
AVAILABLE_MODELS = [
    "qwen/qwen3-32b",
    "qwen/qwen3-14b", 
    "qwen/qwen3-8b",
    "qwen/qwen3-4b",
    "qwen/qwen3-1.7b",
    "qwen/qwen3-0.6b",
]

def run_experiment(recipe, p0_model, p1_model, iteration, experiment_num):
    """Run a single experiment with given parameters and save output to file."""
    cmd = [
        "python", "main.py",
        "--order", recipe,
        "--temperature", "0.7",
        "--p0_gpt_model", p0_model,
        "--p1_gpt_model", p1_model,
        "--p0_local_server_api", "https://openrouter.ai/api/v1",
        "--p1_local_server_api", "https://openrouter.ai/api/v1"
    ]
    
    # Create output directory if it doesn't exist
    output_dir = "experiment_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with timestamp and experiment details
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_p0 = p0_model.replace("/", "_").replace("-", "_")
    safe_p1 = p1_model.replace("/", "_").replace("-", "_")
    filename = f"{output_dir}/exp_{experiment_num:03d}_iter{iteration}_{recipe}_{safe_p0}_vs_{safe_p1}_{timestamp}.log"
    
    print(f"=== Experiment {experiment_num}, Iteration {iteration}, Recipe: {recipe}, P0: {p0_model}, P1: {p1_model} ===")
    print(f"Running: {' '.join(cmd)}")
    print(f"Output will be saved to: {filename}")
    
    # Create initial file with experiment metadata immediately
    def write_initial_file(status="RUNNING"):
        with open(filename, 'w') as f:
            f.write(f"=== Experiment {experiment_num} ===\n")
            f.write(f"Iteration: {iteration}\n")
            f.write(f"Recipe: {recipe}\n")
            f.write(f"P0 Model: {p0_model}\n")
            f.write(f"P1 Model: {p1_model}\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Start Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Status: {status}\n")
            f.write("\n" + "="*50 + "\n")
            f.write("EXPERIMENT IN PROGRESS...\n")
            f.write("This file will be updated when the experiment completes.\n")
            f.write("="*50 + "\n\n")
    
    # Write initial file immediately
    write_initial_file()
    
    try:
        # Start the process without capturing output (so we can stream it)
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Stream output to file in real-time with timeout handling
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Experiment timeout")
        
        stdout_lines = []
        start_time = datetime.now()
        
        # Set up timeout (3600 seconds = 1 hour)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(3600)
        
        try:
            with open(filename, 'a') as f:  # Append mode to add to existing file
                f.write("REAL-TIME OUTPUT:\n")
                f.write("="*50 + "\n")
                f.flush()
                
                # Read output line by line and write to file immediately
                for line in process.stdout:
                    stdout_lines.append(line)
                    f.write(line)
                    f.flush()  # Force write to disk immediately
                    
            # Wait for process to complete and get return code
            return_code = process.wait()
            signal.alarm(0)  # Cancel timeout
            
        except TimeoutError:
            signal.alarm(0)  # Cancel timeout
            print(f"⏰ TIMEOUT: {p0_model} vs {p1_model} on {recipe} → {filename}")
            # Terminate the process
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                
            # Update file with timeout status
            with open(filename, 'a') as f:
                f.write(f"\n{'='*50}\n")
                f.write(f"EXPERIMENT TIMEOUT\n")
                f.write(f"End Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Status: TIMEOUT (exceeded 1 hour limit)\n")
            return False
        
        # Update file with final status
        with open(filename, 'a') as f:
            f.write("\n" + "="*50 + "\n")
            f.write(f"EXPERIMENT COMPLETED\n")
            f.write(f"End Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Exit Code: {return_code}\n")
            f.write(f"Status: {'SUCCESS' if return_code == 0 else 'FAILED'}\n")
        
        if return_code == 0:
            print(f"✓ SUCCESS: {p0_model} vs {p1_model} on {recipe} → {filename}")
            return True
        else:
            print(f"✗ FAILED: {p0_model} vs {p1_model} on {recipe} (exit code: {return_code}) → {filename}")
            return False
            
    except KeyboardInterrupt:
        print(f"🛑 INTERRUPTED: {p0_model} vs {p1_model} on {recipe} → {filename}")
        # Try to terminate the process gracefully
        if 'process' in locals():
            process.terminate()
            try:
                process.wait(timeout=5)  # Wait up to 5 seconds for graceful shutdown
            except subprocess.TimeoutExpired:
                process.kill()  # Force kill if it doesn't terminate gracefully
        
        # Update file with interruption status
        with open(filename, 'a') as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"EXPERIMENT INTERRUPTED\n")
            f.write(f"End Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Status: INTERRUPTED (Ctrl+C pressed)\n")
        
        raise  # Re-raise to allow script termination
        
    except Exception as e:
        print(f"✗ ERROR: {p0_model} vs {p1_model} on {recipe} → {filename}")
        print(f"Exception: {str(e)}")
        
        # Update file with error status
        with open(filename, 'a') as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"EXPERIMENT ERROR\n")
            f.write(f"End Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Status: EXCEPTION\n")
            f.write(f"Error: {str(e)}\n")
        
        return False

def main():
    """Main function to run all combinations."""
    # Check if OpenRouter API key file exists
    openrouter_key_file = "openrouter_key.txt"
    if not os.path.exists(openrouter_key_file):
        print(f"Error: OpenRouter API key file '{openrouter_key_file}' not found!")
        print("Please create this file with your OpenRouter API key.")
        sys.exit(1)
    
    print("Starting model combination experiments...")
    print(f"OpenRouter API key file found: {openrouter_key_file}")
    print(f"Recipes: {len(LEVEL_1_RECIPES)}")
    print(f"Models: {len(AVAILABLE_MODELS)}")
    print(f"Total combinations per iteration: {len(AVAILABLE_MODELS) * len(AVAILABLE_MODELS)}")
    print(f"Total experiments (2 iterations): {2 * len(LEVEL_1_RECIPES) * len(AVAILABLE_MODELS) * len(AVAILABLE_MODELS)}")
    print()
    
    success_count = 0
    failure_count = 0
    experiment_num = 0
    
    # Run experiments 1 times
    for iteration in range(1, 2):
        print(f"\n{'='*50}")
        print(f"STARTING ITERATION {iteration}")
        print(f"{'='*50}")
        
        # For each recipe
        for recipe in LEVEL_1_RECIPES:
            print(f"\n--- Processing recipe: {recipe} (iteration {iteration}) ---")
            
            # For each combination of models
            for p0_model, p1_model in product(AVAILABLE_MODELS, repeat=2):
                experiment_num += 1
                if run_experiment(recipe, p0_model, p1_model, iteration, experiment_num):
                    success_count += 1
                else:
                    failure_count += 1
    
    print(f"\n{'='*50}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*50}")
    print(f"Total experiments: {success_count + failure_count}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failure_count}")
    print(f"Success rate: {success_count/(success_count + failure_count)*100:.1f}%")
    print(f"Output files saved in: experiment_outputs/ directory")

if __name__ == "__main__":
    main()