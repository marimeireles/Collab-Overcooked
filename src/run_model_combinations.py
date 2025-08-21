#!/usr/bin/env python3
"""
Simple script to run all model combinations for level 1 recipes.
Runs each recipe 2x with all possible model combinations.
Based on the user's command template and existing scripts.
"""

import subprocess
import sys
import os
import time
from datetime import datetime
from itertools import product
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import signal

# Level 1 recipes (from convert_result.py)
LEVEL_1_RECIPES = [
    "baked_bell_pepper",
    "baked_sweet_potato", 
    "boiled_egg",
    "boiled_mushroom",
    "boiled_sweet_potato",
]

AVAILABLE_MODELS = [
    "qwen/qwen3-32b",
    "qwen/qwen3-14b",
    "meta-llama/llama-3.3-70b-instruct",
    "google/gemma-3-27b-it",
]

# Rate limiting configuration
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_INTERVAL = 10  # seconds

# Parallelization configuration
MAX_CONCURRENT_EXPERIMENTS = 10

class ThreadSafeRateLimiter:
    """Thread-safe rate limiter to respect API limits across multiple threads."""
    def __init__(self, max_requests=100, time_window=10):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limits. Thread-safe."""
        with self.lock:
            now = time.time()
            
            # Remove requests older than the time window
            while self.requests and self.requests[0] <= now - self.time_window:
                self.requests.popleft()
            
            # If we're at the limit, wait until we can make another request
            if len(self.requests) >= self.max_requests:
                sleep_time = self.requests[0] + self.time_window - now
                if sleep_time > 0:
                    print(f"⏳ Rate limit reached. Waiting {sleep_time:.1f} seconds... (Thread: {threading.current_thread().name})")
                    # Release lock while sleeping to allow other threads to check
                    self.lock.release()
                    time.sleep(sleep_time)
                    self.lock.acquire()
                    # Clean up old requests after waiting
                    while self.requests and self.requests[0] <= time.time() - self.time_window:
                        self.requests.popleft()
            
            # Record this request
            self.requests.append(now)

# Global thread-safe rate limiter instance
rate_limiter = ThreadSafeRateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_INTERVAL)

def run_experiment(recipe, p0_model, p1_model, iteration, experiment_num):
    """Run a single experiment with given parameters and save output to file."""
    # Apply rate limiting before starting the experiment
    rate_limiter.wait_if_needed()
    
    cmd = [
        "python", "main.py",
        "--order", recipe,
        "--temperature", "0.7",
        "--p0_gpt_model", p0_model,
        "--p1_gpt_model", p1_model,
        "--p0_local_server_api", "https://openrouter.ai/api/v1",
        "--p1_local_server_api", "https://openrouter.ai/api/v1",
        "--file_prefix", "cross",
    ]
    
    # Create output directory if it doesn't exist
    output_dir = "experiment_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with timestamp and experiment details
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_p0 = p0_model.replace("/", "_").replace("-", "_")
    safe_p1 = p1_model.replace("/", "_").replace("-", "_")
    filename = f"{output_dir}/cross_exp_{experiment_num:03d}_iter{iteration}_{recipe}_{safe_p0}_vs_{safe_p1}_{timestamp}.log"
    
    thread_name = threading.current_thread().name
    print(f"=== Experiment {experiment_num}, Iteration {iteration}, Recipe: {recipe}, P0: {p0_model}, P1: {p1_model} [Thread: {thread_name}] ===")
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
        def timeout_handler(signum, frame):
            raise TimeoutError("Experiment timeout")
        
        stdout_lines = []
        start_time = datetime.now()
        
        # Set up timeout (3600 seconds = 1 hour)
        # Note: signal only works in main thread, so we'll use a different approach for threads
        timeout_seconds = 3600
        start_time_process = time.time()
        
        try:
            with open(filename, 'a') as f:  # Append mode to add to existing file
                f.write("REAL-TIME OUTPUT:\n")
                f.write("="*50 + "\n")
                f.flush()
                
                # Read output line by line and write to file immediately
                for line in process.stdout:
                    # Check for timeout
                    if time.time() - start_time_process > timeout_seconds:
                        raise TimeoutError("Experiment timeout")
                    
                    stdout_lines.append(line)
                    f.write(line)
                    f.flush()  # Force write to disk immediately
                    
            # Wait for process to complete and get return code
            return_code = process.wait()
            
        except TimeoutError:
            print(f"⏰ TIMEOUT: {p0_model} vs {p1_model} on {recipe} → {filename} (Thread: {threading.current_thread().name})")
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
        
        thread_name = threading.current_thread().name
        if return_code == 0:
            print(f"✓ SUCCESS: {p0_model} vs {p1_model} on {recipe} → {filename} [Thread: {thread_name}]")
            return True
        else:
            print(f"✗ FAILED: {p0_model} vs {p1_model} on {recipe} (exit code: {return_code}) → {filename} [Thread: {thread_name}]")
            return False
            
    except KeyboardInterrupt:
        print(f"🛑 INTERRUPTED: {p0_model} vs {p1_model} on {recipe} → {filename} [Thread: {threading.current_thread().name}]")
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
        print(f"✗ ERROR: {p0_model} vs {p1_model} on {recipe} → {filename} [Thread: {threading.current_thread().name}]")
        print(f"Exception: {str(e)}")
        
        # Update file with error status
        with open(filename, 'a') as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"EXPERIMENT ERROR\n")
            f.write(f"End Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Status: EXCEPTION\n")
            f.write(f"Error: {str(e)}\n")
        
        return False

def create_experiment_tasks():
    """Create all experiment tasks to be run."""
    tasks = []
    experiment_num = 0
    
    # Run experiments 1 time (iteration 1)
    for iteration in range(1, 2):
        # For each recipe
        for recipe in LEVEL_1_RECIPES:
            # For each combination of models
            for p0_model, p1_model in product(AVAILABLE_MODELS, repeat=2):
                experiment_num += 1
                tasks.append((recipe, p0_model, p1_model, iteration, experiment_num))
    
    return tasks

def run_parallel_experiments():
    """Run experiments in parallel using ThreadPoolExecutor."""
    tasks = create_experiment_tasks()
    total_tasks = len(tasks)
    
    print(f"\n🚀 Starting parallel execution with {MAX_CONCURRENT_EXPERIMENTS} concurrent workers")
    print(f"📋 Total tasks to process: {total_tasks}")
    print(f"⚡ This should be approximately {MAX_CONCURRENT_EXPERIMENTS}x faster than sequential execution!")
    print()
    
    success_count = 0
    failure_count = 0
    completed_count = 0
    
    start_time = time.time()
    
    # Use ThreadPoolExecutor to run experiments in parallel
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_EXPERIMENTS, thread_name_prefix="ExpWorker") as executor:
        # Submit all tasks
        future_to_task = {executor.submit(run_experiment, *task): task for task in tasks}
        
        try:
            # Process completed futures as they finish
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                completed_count += 1
                
                try:
                    result = future.result()
                    if result:
                        success_count += 1
                    else:
                        failure_count += 1
                except Exception as e:
                    failure_count += 1
                    recipe, p0_model, p1_model, iteration, experiment_num = task
                    print(f"✗ EXCEPTION in task {experiment_num}: {p0_model} vs {p1_model} on {recipe} - {str(e)}")
                
                # Progress update
                elapsed_time = time.time() - start_time
                progress_pct = (completed_count / total_tasks) * 100
                avg_time_per_task = elapsed_time / completed_count if completed_count > 0 else 0
                remaining_tasks = total_tasks - completed_count
                eta_seconds = remaining_tasks * avg_time_per_task / MAX_CONCURRENT_EXPERIMENTS
                
                print(f"📊 Progress: {completed_count}/{total_tasks} ({progress_pct:.1f}%) | "
                      f"✅ {success_count} success, ❌ {failure_count} failed | "
                      f"⏱️  ETA: {eta_seconds/60:.1f}min")
                
        except KeyboardInterrupt:
            print("\n🛑 Received interrupt signal. Shutting down workers...")
            # Cancel all pending futures
            for future in future_to_task:
                future.cancel()
            raise
    
    return success_count, failure_count, completed_count

def main():
    """Main function to run all combinations in parallel."""
    # Check if OpenRouter API key file exists
    openrouter_key_file = "openrouter_key.txt"
    if not os.path.exists(openrouter_key_file):
        print(f"Error: OpenRouter API key file '{openrouter_key_file}' not found!")
        print("Please create this file with your OpenRouter API key.")
        sys.exit(1)
    
    print("🚀 Starting PARALLEL model combination experiments...")
    print(f"OpenRouter API key file found: {openrouter_key_file}")
    print(f"Rate limiting: {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_INTERVAL} seconds (shared across all threads)")
    print(f"Parallel workers: {MAX_CONCURRENT_EXPERIMENTS}")
    print(f"Recipes: {len(LEVEL_1_RECIPES)}")
    print(f"Models: {len(AVAILABLE_MODELS)}")
    print(f"Total combinations per iteration: {len(AVAILABLE_MODELS) * len(AVAILABLE_MODELS)}")
    
    total_experiments = 1 * len(LEVEL_1_RECIPES) * len(AVAILABLE_MODELS) * len(AVAILABLE_MODELS)
    print(f"Total experiments (1 iteration): {total_experiments}")
    
    # Estimate time with parallelization
    sequential_time_minutes = (total_experiments / RATE_LIMIT_REQUESTS) * (RATE_LIMIT_INTERVAL / 60)
    parallel_time_minutes = sequential_time_minutes / MAX_CONCURRENT_EXPERIMENTS
    print(f"Estimated time (sequential): {sequential_time_minutes:.1f} minutes")
    print(f"Estimated time (parallel): {parallel_time_minutes:.1f} minutes")
    print(f"⚡ Speed improvement: ~{MAX_CONCURRENT_EXPERIMENTS}x faster!")
    print()
    
    try:
        success_count, failure_count, completed_count = run_parallel_experiments()
        
        print(f"\n{'='*50}")
        print("🎯 EXPERIMENT SUMMARY")
        print(f"{'='*50}")
        print(f"Total experiments: {completed_count}")
        print(f"✅ Successful: {success_count}")
        print(f"❌ Failed: {failure_count}")
        if completed_count > 0:
            print(f"📈 Success rate: {success_count/completed_count*100:.1f}%")
        print(f"📁 Output files saved in: experiment_outputs/ directory")
        
    except KeyboardInterrupt:
        print(f"\n{'='*50}")
        print("🛑 EXPERIMENT INTERRUPTED BY USER")
        print(f"{'='*50}")
        print("Partial results may be available in: experiment_outputs/ directory")
        sys.exit(1)

if __name__ == "__main__":
    # Handle Ctrl+C gracefully
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)