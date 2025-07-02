#!/bin/bash
#SBATCH --job-name=qwen_llama_overcooked_matrix
#SBATCH --output=slurm/%x_%j.log
#SBATCH --cpus-per-task=16
#SBATCH --mem=48GB
#SBATCH --gres=gpu:A100-PCI-80GB:2
#SBATCH --time=6:00:00
#SBATCH --nodelist=rlhf.ist.berkeley.edu


set -euo pipefail
set -a
source /nas/ucb/marimeireles/dev/Collab-Overcooked/slurm-scripts/secrets.env
set +a

echo "Running on host: $(hostname)"

###############################################################################
# 1)  Metadata for model matrix test
###############################################################################
# Model directories
dir_qwen_7b="/nas/ucb/marimeireles/cache/hub/models--Qwen--Qwen2.5-7B-Instruct"
dir_llama_7b="/nas/ucb/marimeireles/cache/hub/models--meta-llama--Meta-Llama-3-8B-Instruct"
dir_llama_14b="/nas/ucb/marimeireles/cache/hub/models--meta-llama--Meta-Llama-3-70B-Instruct"

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
    pgrep -f "Llama.*Instruct" | xargs -r kill -9 2>/dev/null || true
    pgrep -f "Meta-Llama" | xargs -r kill -9 2>/dev/null || true
    
    # Kill processes on all our ports (both 8xxx and 4xxx ranges)
    for port in 8070 8071 4070 4071 4072 4073; do
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
    for port in 8070 8071 4070 4071 4072 4073; do
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

# Port configuration for all models
qwen_7b_p0_port="http://localhost:8070/v1"
qwen_7b_p1_port="http://localhost:8071/v1"
llama_7b_p0_port="http://localhost:4070/v1"
llama_7b_p1_port="http://localhost:4071/v1"
llama_14b_p0_port="http://localhost:4072/v1"
llama_14b_p1_port="http://localhost:4073/v1"

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

# Function to start a model server
start_model_server() {
    local model_name=$1
    local model_dir=$2
    local port=$3
    local gpu_util=$4
    local log_suffix=$5
    
    local port_num=${port#http://localhost:}
    port_num=${port_num%/v1}
    
    echo "Starting $model_name on port $port_num..."
    vllm serve "$model_name" \
        --tensor-parallel-size 1 \
        --host 0.0.0.0 \
        --port $port_num \
        --trust-remote-code \
        --temperature 0.7 \
        --gpu-memory-utilization $gpu_util > "/nas/ucb/$USER/dev/Collab-Overcooked/slurm-scripts/slurm/vllm_$log_suffix.log" 2>&1 &
    
    echo $!
}

# Function to run experiment
run_experiment() {
    local exp_name=$1
    local p0_model=$2
    local p1_model=$3
    local p0_dir=$4
    local p1_dir=$5
    local p0_port=$6
    local p1_port=$7
    
    echo "=== Running experiment: $exp_name ==="
    
    # Run the experiment with 60-minute timeout
    echo "Running experiment $exp_name... (60-minute timeout)"
    if timeout 3600 srun --nodes=1 --ntasks=1 python main.py \
        --order boiled_egg \
        --p0_gpt_model "$p0_model" \
        --p1_gpt_model "$p1_model" \
        --p0_model_dirname "$p0_dir" \
        --p1_model_dirname "$p1_dir" \
        --p0_local_server_api "$p0_port" \
        --p1_local_server_api "$p1_port"; then
        echo "Experiment $exp_name completed successfully."
        return 0
    else
        exit_code=$?
        if [ $exit_code -eq 124 ]; then
            echo "WARNING: Experiment $exp_name timed out after 60 minutes."
        else
            echo "WARNING: Experiment $exp_name failed with exit code $exit_code."
        fi
        return $exit_code
    fi
}

# Check GPU memory before starting
check_gpu_memory

echo "=== Starting model matrix experiments ==="

###############################################################################
# Experiment 1: Qwen 7B vs Qwen 7B
###############################################################################
echo "Starting Qwen 7B servers..."
p0_pid=$(start_model_server "Qwen/Qwen2.5-7B-Instruct" "$dir_qwen_7b" "$qwen_7b_p0_port" "0.35" "qwen7b_p0")
p1_pid=$(start_model_server "Qwen/Qwen2.5-7B-Instruct" "$dir_qwen_7b" "$qwen_7b_p1_port" "0.35" "qwen7b_p1")

# Wait for servers to be ready
if ! check_server_ready "$qwen_7b_p0_port" "Qwen/Qwen2.5-7B-Instruct"; then
    echo "Failed to start Qwen 7B server (p0). Skipping experiment."
    kill $p0_pid $p1_pid 2>/dev/null || true
else
    if ! check_server_ready "$qwen_7b_p1_port" "Qwen/Qwen2.5-7B-Instruct"; then
        echo "Failed to start Qwen 7B server (p1). Skipping experiment."
        kill $p0_pid $p1_pid 2>/dev/null || true
    else
        run_experiment "Qwen7B_vs_Qwen7B" \
            "Qwen/Qwen2.5-7B-Instruct" "Qwen/Qwen2.5-7B-Instruct" \
            "$dir_qwen_7b" "$dir_qwen_7b" \
            "$qwen_7b_p0_port" "$qwen_7b_p1_port"
    fi
fi

# Stop Qwen servers
echo "Stopping Qwen 7B servers..."
kill -TERM $p0_pid $p1_pid 2>/dev/null || true
sleep 10
kill -9 $p0_pid $p1_pid 2>/dev/null || true

###############################################################################
# Experiment 2: Llama 7B vs Llama 7B  
###############################################################################
echo "Starting Llama 7B servers..."
p0_pid=$(start_model_server "meta-llama/Meta-Llama-3-8B-Instruct" "$dir_llama_7b" "$llama_7b_p0_port" "0.35" "llama7b_p0")
p1_pid=$(start_model_server "meta-llama/Meta-Llama-3-8B-Instruct" "$dir_llama_7b" "$llama_7b_p1_port" "0.35" "llama7b_p1")

# Wait for servers to be ready
if ! check_server_ready "$llama_7b_p0_port" "meta-llama/Meta-Llama-3-8B-Instruct"; then
    echo "Failed to start Llama 7B server (p0). Skipping experiment."
    kill $p0_pid $p1_pid 2>/dev/null || true
else
    if ! check_server_ready "$llama_7b_p1_port" "meta-llama/Meta-Llama-3-8B-Instruct"; then
        echo "Failed to start Llama 7B server (p1). Skipping experiment."
        kill $p0_pid $p1_pid 2>/dev/null || true
    else
        run_experiment "Llama7B_vs_Llama7B" \
            "meta-llama/Meta-Llama-3-8B-Instruct" "meta-llama/Meta-Llama-3-8B-Instruct" \
            "$dir_llama_7b" "$dir_llama_7b" \
            "$llama_7b_p0_port" "$llama_7b_p1_port"
    fi
fi

# Stop Llama 7B servers
echo "Stopping Llama 7B servers..."
kill -TERM $p0_pid $p1_pid 2>/dev/null || true
sleep 10
kill -9 $p0_pid $p1_pid 2>/dev/null || true

###############################################################################
# Experiment 3: Llama 14B vs Llama 14B
###############################################################################
echo "Starting Llama 14B servers..."
p0_pid=$(start_model_server "meta-llama/Meta-Llama-3-70B-Instruct" "$dir_llama_14b" "$llama_14b_p0_port" "0.8" "llama14b_p0")
p1_pid=$(start_model_server "meta-llama/Meta-Llama-3-70B-Instruct" "$dir_llama_14b" "$llama_14b_p1_port" "0.8" "llama14b_p1")

# Wait for servers to be ready
if ! check_server_ready "$llama_14b_p0_port" "meta-llama/Meta-Llama-3-70B-Instruct"; then
    echo "Failed to start Llama 14B server (p0). Skipping experiment."
    kill $p0_pid $p1_pid 2>/dev/null || true
else
    if ! check_server_ready "$llama_14b_p1_port" "meta-llama/Meta-Llama-3-70B-Instruct"; then
        echo "Failed to start Llama 14B server (p1). Skipping experiment."
        kill $p0_pid $p1_pid 2>/dev/null || true
    else
        run_experiment "Llama14B_vs_Llama14B" \
            "meta-llama/Meta-Llama-3-70B-Instruct" "meta-llama/Meta-Llama-3-70B-Instruct" \
            "$dir_llama_14b" "$dir_llama_14b" \
            "$llama_14b_p0_port" "$llama_14b_p1_port"
    fi
fi

# Stop Llama 14B servers
echo "Stopping Llama 14B servers..."
kill -TERM $p0_pid $p1_pid 2>/dev/null || true
sleep 10
kill -9 $p0_pid $p1_pid 2>/dev/null || true

# Wait for processes to clean up and ensure ports are freed
echo "Waiting for processes to terminate and ports to be freed..."
sleep 20

# Force kill any remaining vLLM processes on all ports
for port in 8070 8071 4070 4071 4072 4073; do
    for attempt in 1 2 3; do
        echo "Port $port cleanup attempt $attempt..."
        lsof -ti:$port | xargs -r kill -9 2>/dev/null || true
        sleep 5
    done
done

# Additional cleanup - kill any vLLM processes that might still be running
pkill -f "vllm serve" 2>/dev/null || true
pkill -f "Qwen.*Instruct" 2>/dev/null || true
pkill -f "Meta-Llama" 2>/dev/null || true

# Clear GPU memory
echo "Clearing GPU memory..."
nvidia-smi --gpu-reset 2>/dev/null || true
python3 -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print('GPU cache cleared')
" 2>/dev/null || true

# Additional wait to ensure ports are fully released
sleep 15

# Verify ports are free
for port in 8070 8071 4070 4071 4072 4073; do
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

# Final cleanup
echo "=== Final cleanup ==="
cleanup_existing_servers
echo "Model matrix experiments completed!"
