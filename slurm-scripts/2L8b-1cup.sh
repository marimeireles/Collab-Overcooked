#!/bin/bash
#SBATCH --job-name=l8b_1cup
#SBATCH --output=slurm/%j.log
#SBATCH --cpus-per-task=16
#SBATCH --mem=96GB
#SBATCH --gres=gpu:A100-PCI-80GB:1
#SBATCH --time=12:00:00
#SBATCH --nodelist=cirl.ist.berkeley.edu

set -euo pipefail
set -a
source /nas/ucb/marimeireles/dev/Collab-Overcooked/slurm-scripts/secrets.env
set +a

echo "Running on host: $(hostname)"

# 1) Point to micromamba root
export MAMBA_ROOT_PREFIX=/nas/ucb/$USER/micromamba
export PATH=$MAMBA_ROOT_PREFIX/bin:$PATH

# 2) Export environment variables
export TMPDIR=/nas/ucb/$USER/pip_tmp
export HUGGINGFACE_HUB_CACHE="/nas/ucb/marimeireles/cache/hub"
export HF_HOME="/nas/ucb/marimeireles/cache/hub"
export XDG_CONFIG_HOME="/nas/ucb/marimeireles/.config"
export OUTLINES_CACHE_DIR="/nas/ucb/marimeireles/cache/outlines"

# 3) Initialize micromamba shell environment
eval "$($MAMBA_ROOT_PREFIX/micromamba shell hook --shell bash)"

# 4) Activate environment
micromamba activate $MAMBA_ROOT_PREFIX/envs/overcooked

# 5) Change to project directory
cd /nas/ucb/$USER/dev/Collab-Overcooked

# 6) Login and download model
huggingface-cli login --token "$HF_TOKEN"
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct

# Function to check if a server is ready
check_server_ready() {
    local port=$1
    local model_name=$2
    local max_attempts=30
    local attempt=1
    
    echo "Waiting for server on port $port to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -X POST "http://localhost:$port/v1/completions" \
            -H "Content-Type: application/json" \
            -d "{
                \"model\": \"$model_name\",
                \"prompt\": \"test\",
                \"max_tokens\": 1,
                \"temperature\": 0.7
            }" > /dev/null 2>&1; then
            echo "Server on port $port is ready!"
            return 0
        fi
        
        echo "Attempt $attempt/$max_attempts: Server not ready yet, waiting 10 seconds..."
        sleep 10
        attempt=$((attempt + 1))
    done
    
    echo "ERROR: Server on port $port failed to start after $max_attempts attempts"
    return 1
}

# 7) Launch first vLLM server on port 4140 (GPU 0)
echo "Starting first Llama model on port 4140 (GPU 0)..."
CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
       --host 0.0.0.0 \
       --port 4140 \
       --trust-remote-code \
       --gpu-memory-utilization 0.4 > "/nas/ucb/$USER/dev/Collab-Overcooked/slurm-scripts/slurm/vllm_8140.log" 2>&1 &
server1_pid=$!

# Wait for first server to be ready
if ! check_server_ready "4140" "meta-llama/Meta-Llama-3-8B-Instruct"; then
    echo "Failed to start server on port 4140. Exiting."
    kill $server1_pid 2>/dev/null || true
    exit 1
fi

# 8) Launch second vLLM server on port 4141 (GPU 0 - same as first)
echo "Starting second Llama model on port 4141 (GPU 0)..."
echo "Available GPUs: $(nvidia-smi -L)"
echo "GPU 0 status: $(nvidia-smi -i 0 --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits)"
CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
       --host 0.0.0.0 \
       --port 4141 \
       --trust-remote-code \
       --gpu-memory-utilization 0.4 > "/nas/ucb/$USER/dev/Collab-Overcooked/slurm-scripts/slurm/vllm_8141.log" 2>&1 &
server2_pid=$!

# Wait for second server to be ready
if ! check_server_ready "4141" "meta-llama/Meta-Llama-3-8B-Instruct"; then
    echo "Failed to start server on port 4141. Exiting."
    kill $server1_pid 2>/dev/null || true
    kill $server2_pid 2>/dev/null || true
    exit 1
fi

echo "Both servers are running and ready!"
echo "Server 1: http://localhost:4140/v1"
echo "Server 2: http://localhost:4141/v1"

# Keep the script running to maintain the servers
echo "Servers are running. Press Ctrl+C to stop them."
trap 'echo "Stopping servers..."; kill $server1_pid $server2_pid 2>/dev/null || true; exit 0' INT TERM

# Wait for either server to finish
wait


