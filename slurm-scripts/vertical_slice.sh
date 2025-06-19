#!/bin/bash
#SBATCH --job-name=qwen_overcooked_matrix
#SBATCH --output=slurm/%x_%j.log
#SBATCH --cpus-per-task=16
#SBATCH --mem=96GB
#SBATCH --gres=gpu:A100-PCI-80GB:4
#SBATCH --time=6:00:00
#SBATCH --nodelist=rlhf.ist.berkeley.edu


set -euo pipefail
set -a
source /nas/ucb/marimeireles/dev/Collab-Overcooked/slurm-scripts/secrets.env
set +a

echo "Running on host: $(hostname)"

###############################################################################
# 1)  Metadata that never changes across the two matrices
###############################################################################
models=(
  "Qwen/Qwen2.5-14B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct"
)

ports_0=(
  "http://localhost:8140/v1"
  "http://localhost:8070/v1"
)

ports_1=(
  "http://localhost:8141/v1"
  "http://localhost:8071/v1"
)

dirnames=(
  "/nas/ucb/marimeireles/cache/hub/models--Qwen--Qwen2.5-14B-Instruct"
  "/nas/ucb/marimeireles/cache/hub/models--Qwen--Qwen2.5-7B-Instruct"
)

recipe_dir="/nas/ucb/marimeireles/dev/Collab-Overcooked/src/prompts/recipe"

###############################################################################
# 2)  Environment bootstrap
###############################################################################
export MAMBA_ROOT_PREFIX=/nas/ucb/$USER/micromamba
export PATH=$MAMBA_ROOT_PREFIX/bin:$PATH
export TMPDIR=/nas/ucb/$USER/pip_tmp
export HUGGINGFACE_HUB_CACHE="/nas/ucb/marimeireles/cache/hub"
export HF_HOME="/nas/ucb/marimeireles/cache/hub"
export XDG_CONFIG_HOME="/nas/ucb/marimeireles/.config"
export OUTLINES_CACHE_DIR="/nas/ucb/marimeireles/cache/outlines"

eval "$($MAMBA_ROOT_PREFIX/micromamba shell hook --shell bash)"
micromamba activate "$MAMBA_ROOT_PREFIX/envs/overcooked"

cd /nas/ucb/$USER/dev/Collab-Overcooked/src

# Function to clean up any existing vLLM processes
cleanup_existing_servers() {
    echo "Cleaning up any existing vLLM processes..."
    
    # Kill all vLLM processes more aggressively
    pkill -f "vllm serve" 2>/dev/null || true
    pkill -f "vllm.entrypoints" 2>/dev/null || true
    pkill -f "python.*vllm" 2>/dev/null || true
    
    # Kill any Python processes that might be vLLM servers
    pgrep -f "Qwen.*Instruct" | xargs -r kill -9 2>/dev/null || true
    
    # Kill processes on our specific ports
    for port in 8140 8141 8070 8071; do
        if lsof -ti:$port >/dev/null 2>&1; then
            echo "Killing processes on port $port..."
            lsof -ti:$port | xargs -r kill -9 2>/dev/null || true
            sleep 2
            # Double check and kill again if needed
            if lsof -ti:$port >/dev/null 2>&1; then
                echo "Port $port still occupied, trying again..."
                lsof -ti:$port | xargs -r kill -9 2>/dev/null || true
            fi
        fi
    done
    
    # Clear GPU memory using nvidia-smi
    echo "Clearing GPU memory caches..."
    nvidia-smi --gpu-reset 2>/dev/null || true
    
    # Clear PyTorch CUDA cache by running a small Python script
    python3 -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print('GPU cache cleared')
" 2>/dev/null || true
    
    # Wait for cleanup
    sleep 15
    
    # Verify all our ports are free
    for port in 8140 8141 8070 8071; do
        if lsof -ti:$port >/dev/null 2>&1; then
            echo "WARNING: Port $port is still in use after cleanup!"
            # Show what's using the port
            lsof -i:$port || true
        else
            echo "Port $port is free"
        fi
    done
    
    # Show GPU memory status
    echo "GPU memory status after cleanup:"
    nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total --format=csv
}

# Clean up before starting
cleanup_existing_servers

# Port configuration
# When 14B is p0, use 8140, when 14B is p1, use 8141
# When 7B is p0, use 8070, when 7B is p1, use 8071
p0_14b_port="http://localhost:8140/v1"
p1_14b_port="http://localhost:8141/v1"
p0_7b_port="http://localhost:8070/v1"
p1_7b_port="http://localhost:8071/v1"

# Model directories
dir_14b="/nas/ucb/marimeireles/cache/hub/models--Qwen--Qwen2.5-14B-Instruct"
dir_7b="/nas/ucb/marimeireles/cache/hub/models--Qwen--Qwen2.5-7B-Instruct"

# Function to check if a server is ready
check_server_ready() {
    local port=$1
    local model_name=$2
    local max_attempts=30
    local attempt=1
    
    # Extract just the port number
    local port_num=${port#http://localhost:}
    port_num=${port_num%/v1}
    
    echo "Waiting for server on port $port_num to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -X POST "http://localhost:$port_num/v1/completions" \
            -H "Content-Type: application/json" \
            -d "{
                \"model\": \"$model_name\",
                \"prompt\": \"test\",
                \"max_tokens\": 1,
                \"temperature\": 0.7
            }" > /dev/null 2>&1; then
            echo "Server on port $port_num is ready!"
            return 0
        fi
        
        echo "Attempt $attempt/$max_attempts: Server not ready yet, waiting 10 seconds..."
        sleep 10
        attempt=$((attempt + 1))
    done
    
    echo "ERROR: Server on port $port_num failed to start after $max_attempts attempts"
    return 1
}

# Function to check GPU memory availability
check_gpu_memory() {
    echo "Checking GPU memory availability..."
    nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total --format=csv,noheader,nounits | while read line; do
        gpu_id=$(echo $line | cut -d',' -f1)
        memory_used=$(echo $line | cut -d',' -f2)
        memory_free=$(echo $line | cut -d',' -f3)
        memory_total=$(echo $line | cut -d',' -f4)
        echo "GPU $gpu_id: Used=${memory_used}MB, Free=${memory_free}MB, Total=${memory_total}MB"
        
        # Check if we have at least 30GB free (for safety margin)
        if [ $memory_free -lt 30000 ]; then
            echo "WARNING: GPU $gpu_id has less than 30GB free memory!"
        fi
    done
}

# Model combinations to test
combinations=(
    "14B 7B"    # 14B as p0, 7B as p1
    "14B 14B"   # 14B as p0, 14B as p1
    "7B 14B"    # 7B as p0, 14B as p1
    "7B 7B"     # 7B as p0, 7B as p1
)

# Run the experiment matrix 20 times
for run_iteration in {1..20}; do
    echo "=== Starting run iteration $run_iteration/20 ==="
    
    # Loop through combinations
for combo in "${combinations[@]}"; do
    # Split the combination into p0 and p1 models
    read -r p0_model p1_model <<< "$combo"
    
    # Set ports based on model sizes
    if [ "$p0_model" = "14B" ]; then
        p0_port="$p0_14b_port"
        p0_dir="$dir_14b"
    else
        p0_port="$p0_7b_port"
        p0_dir="$dir_7b"
    fi
    
    if [ "$p1_model" = "14B" ]; then
        p1_port="$p1_14b_port"
        p1_dir="$dir_14b"
    else
        p1_port="$p1_7b_port"
        p1_dir="$dir_7b"
    fi
    
    echo "=== Starting experiment ${run_iteration}/20: ${p0_model} (p0) vs ${p1_model} (p1) ==="
    
    # Check GPU memory before starting
    check_gpu_memory
    
    # Start both models sequentially to avoid resource conflicts
    
    # Start first model (p0)
    p0_port_num=${p0_port#http://localhost:}
    p0_port_num=${p0_port_num%/v1}
    echo "Starting ${p0_model} model on port $p0_port_num..."
    vllm serve "Qwen/Qwen2.5-${p0_model}-Instruct" \
        --tensor-parallel-size 2 \
        --host 0.0.0.0 \
        --port $p0_port_num \
        --trust-remote-code \
        --gpu-memory-utilization 0.35 > "/nas/ucb/$USER/dev/Collab-Overcooked/slurm-scripts/slurm/vllm_${p0_model}_p0.log" 2>&1 &
    p0_pid=$!

    # Wait for first server to be ready
    if ! check_server_ready "$p0_port" "Qwen/Qwen2.5-${p0_model}-Instruct"; then
        echo "Failed to start ${p0_model} server. Exiting."
        kill $p0_pid 2>/dev/null || true
        exit 1
    fi

    # Check GPU memory before starting second model
    echo "GPU memory status before starting second model:"
    check_gpu_memory

    # Start second model (p1)
    p1_port_num=${p1_port#http://localhost:}
    p1_port_num=${p1_port_num%/v1}
    echo "Starting ${p1_model} model on port $p1_port_num..."
    vllm serve "Qwen/Qwen2.5-${p1_model}-Instruct" \
        --tensor-parallel-size 2 \
        --host 0.0.0.0 \
        --port $p1_port_num \
        --trust-remote-code \
        --gpu-memory-utilization 0.35 > "/nas/ucb/$USER/dev/Collab-Overcooked/slurm-scripts/slurm/vllm_${p1_model}_p1.log" 2>&1 &
    p1_pid=$!

    # Wait for second server to be ready
    if ! check_server_ready "$p1_port" "Qwen/Qwen2.5-${p1_model}-Instruct"; then
        echo "Failed to start ${p1_model} server. Exiting."
        kill $p0_pid 2>/dev/null || true
        kill $p1_pid 2>/dev/null || true
        exit 1
    fi

    # Run the experiment with 60-minute timeout
    echo "Running experiment with ${p0_model} (p0) and ${p1_model} (p1)... (60-minute timeout)"
    if timeout 3600 srun --nodes=1 --ntasks=1 python main.py \
        --order boiled_egg \
        --p0_gpt_model "Qwen/Qwen2.5-${p0_model}-Instruct" \
        --p1_gpt_model "Qwen/Qwen2.5-${p1_model}-Instruct" \
        --p0_model_dirname "$p0_dir" \
        --p1_model_dirname "$p1_dir" \
        --p0_local_server_api "$p0_port" \
        --p1_local_server_api "$p1_port"; then
        echo "Experiment completed successfully."
    else
        exit_code=$?
        if [ $exit_code -eq 124 ]; then
            echo "WARNING: Experiment timed out after 60 minutes. Moving to next iteration."
        else
            echo "WARNING: Experiment failed with exit code $exit_code. Moving to next iteration."
        fi
    fi
    
    # Kill both models
    echo "Stopping ${p0_model} model (PID: $p0_pid)..."
    kill -TERM $p0_pid 2>/dev/null || true
    echo "Stopping ${p1_model} model (PID: $p1_pid)..."
    kill -TERM $p1_pid 2>/dev/null || true

    # Wait for graceful shutdown
    sleep 10
    
    # Force kill if still running
    kill -9 $p0_pid 2>/dev/null || true
    kill -9 $p1_pid 2>/dev/null || true
    
    # Kill any child processes that might still be running
    pkill -P $p0_pid 2>/dev/null || true
    pkill -P $p1_pid 2>/dev/null || true

    # Wait for processes to clean up and ensure ports are freed
    echo "Waiting for processes to terminate and ports to be freed..."
    sleep 20

    # Force kill any remaining vLLM processes on the specific ports
    p0_port_num=${p0_port#http://localhost:}
    p0_port_num=${p0_port_num%/v1}
    p1_port_num=${p1_port#http://localhost:}  
    p1_port_num=${p1_port_num%/v1}

    # Kill any process using these ports (multiple attempts)
    for attempt in 1 2 3; do
        echo "Port cleanup attempt $attempt..."
        lsof -ti:$p0_port_num | xargs -r kill -9 2>/dev/null || true
        lsof -ti:$p1_port_num | xargs -r kill -9 2>/dev/null || true
        sleep 5
    done

    # Additional cleanup - kill any vLLM processes that might still be running
    pkill -f "vllm serve" 2>/dev/null || true
    pkill -f "Qwen.*Instruct" 2>/dev/null || true

    # Clear GPU memory between experiments
    echo "Clearing GPU memory between experiments..."
    nvidia-smi --gpu-reset 2>/dev/null || true
    python3 -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print('GPU cache cleared between experiments')
" 2>/dev/null || true

    # Additional wait to ensure ports are fully released
    sleep 15

    # Verify ports are free before continuing
    for port in $p0_port_num $p1_port_num; do
        max_attempts=6
        attempt=1
        while lsof -ti:$port >/dev/null 2>&1 && [ $attempt -le $max_attempts ]; do
            echo "Port $port still in use (attempt $attempt/$max_attempts), killing remaining processes..."
            lsof -ti:$port | xargs -r kill -9 2>/dev/null || true
            sleep 10
            attempt=$((attempt + 1))
        done
        
        if lsof -ti:$port >/dev/null 2>&1; then
            echo "ERROR: Port $port is still in use after $max_attempts attempts!"
            lsof -i:$port || true
        else
            echo "Port $port is now free"
        fi
    done
done

echo "=== Completed run iteration $run_iteration/20 ==="
done

# Final cleanup
echo "=== Final cleanup ==="
cleanup_existing_servers
echo "All experiments completed!"