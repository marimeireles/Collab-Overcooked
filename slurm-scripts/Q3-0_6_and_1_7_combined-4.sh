#!/bin/bash
#SBATCH --job-name=Q3-06-17-4
#SBATCH --output=slurm/%j.log
#SBATCH --cpus-per-task=16
#SBATCH --mem=48GB
#SBATCH --gres=gpu:A100-SXM4-80GB:1
#SBATCH --time=48:00:00
#SBATCH --nodelist=sac.ist.berkeley.edu

set -euo pipefail
set -a
source /nas/ucb/marimeireles/dev/Collab-Overcooked/slurm-scripts/secrets.env
set +a

echo "Running on host: $(hostname)"

# 1) Point to micromamba root
export MAMBA_ROOT_PREFIX=/nas/ucb/marimeireles/micromamba
export PATH=$MAMBA_ROOT_PREFIX/bin:$PATH

# 2) Export environment variables
export TMPDIR=/nas/ucb/marimeireles/pip_tmp
export HUGGINGFACE_HUB_CACHE="/nas/ucb/marimeireles/cache/hub"
export HF_HOME="/nas/ucb/marimeireles/cache/hub"
export XDG_CONFIG_HOME="/nas/ucb/marimeireles/.config"
export OUTLINES_CACHE_DIR="/nas/ucb/marimeireles/cache/outlines"

# 3) Initialize micromamba shell environment
eval "$($MAMBA_ROOT_PREFIX/micromamba shell hook --shell bash)"

# 4) Activate environment
micromamba activate $MAMBA_ROOT_PREFIX/envs/overcooked

# 5) Change to project directory
cd /nas/ucb/marimeireles/dev/Collab-Overcooked

# 6) Login and download models
huggingface-cli login --token "$HF_TOKEN"
huggingface-cli download Qwen/Qwen3-0.6B
huggingface-cli download Qwen/Qwen3-1.7B

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

# 7) Launch first Qwen3-0.6B server on port 8006 (GPU 0)
echo "Starting first Qwen3-0.6B model on port 8006 (GPU 0)..."
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-0.6B \
       --host 0.0.0.0 \
       --port 8006 \
       --trust-remote-code \
       --gpu-memory-utilization 0.2 \
       --max-model-len 8192 > "/nas/ucb/marimeireles/dev/Collab-Overcooked/slurm-scripts/slurm/vllm_8006.log" 2>&1 &
server1_pid=$!

# Wait for first server to be ready
if ! check_server_ready "8006" "Qwen/Qwen3-0.6B"; then
    echo "Failed to start server on port 8006. Exiting."
    kill $server1_pid 2>/dev/null || true
    exit 1
fi

# 8) Launch second Qwen3-0.6B server on port 8061 (GPU 0)
echo "Starting second Qwen3-0.6B model on port 8061 (GPU 0)..."
echo "Available GPUs: $(nvidia-smi -L)"
echo "GPU 0 status: $(nvidia-smi -i 0 --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits)"
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-0.6B \
       --host 0.0.0.0 \
       --port 8061 \
       --trust-remote-code \
       --gpu-memory-utilization 0.2 \
       --max-model-len 8192 > "/nas/ucb/marimeireles/dev/Collab-Overcooked/slurm-scripts/slurm/vllm_8061.log" 2>&1 &
server2_pid=$!

# Wait for second server to be ready
if ! check_server_ready "8061" "Qwen/Qwen3-0.6B"; then
    echo "Failed to start server on port 8061. Exiting."
    kill $server1_pid 2>/dev/null || true
    kill $server2_pid 2>/dev/null || true
    exit 1
fi

# 9) Launch first Qwen3-1.7B server on port 8170 (GPU 0)
echo "Starting first Qwen3-1.7B model on port 8170 (GPU 0)..."
echo "GPU 0 status: $(nvidia-smi -i 0 --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits)"
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-1.7B \
       --host 0.0.0.0 \
       --port 8170 \
       --trust-remote-code \
       --gpu-memory-utilization 0.2 \
       --max-model-len 8192 > "/nas/ucb/marimeireles/dev/Collab-Overcooked/slurm-scripts/slurm/vllm_8170.log" 2>&1 &
server3_pid=$!

# Wait for third server to be ready
if ! check_server_ready "8170" "Qwen/Qwen3-1.7B"; then
    echo "Failed to start server on port 8170. Exiting."
    kill $server1_pid 2>/dev/null || true
    kill $server2_pid 2>/dev/null || true
    kill $server3_pid 2>/dev/null || true
    exit 1
fi

# 10) Launch second Qwen3-1.7B server on port 8171 (GPU 0)
echo "Starting second Qwen3-1.7B model on port 8171 (GPU 0)..."
echo "GPU 0 status: $(nvidia-smi -i 0 --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits)"
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-1.7B \
       --host 0.0.0.0 \
       --port 8171 \
       --trust-remote-code \
       --gpu-memory-utilization 0.2 \
       --max-model-len 8192 > "/nas/ucb/marimeireles/dev/Collab-Overcooked/slurm-scripts/slurm/vllm_8171.log" 2>&1 &
server4_pid=$!

# Wait for fourth server to be ready
if ! check_server_ready "8171" "Qwen/Qwen3-1.7B"; then
    echo "Failed to start server on port 8171. Exiting."
    kill $server1_pid 2>/dev/null || true
    kill $server2_pid 2>/dev/null || true
    kill $server3_pid 2>/dev/null || true
    kill $server4_pid 2>/dev/null || true
    exit 1
fi

echo "All four servers are running and ready!"
echo "Qwen3-0.6B Server 1: http://localhost:8006/v1"
echo "Qwen3-0.6B Server 2: http://localhost:8061/v1"
echo "Qwen3-1.7B Server 1: http://localhost:8170/v1"
echo "Qwen3-1.7B Server 2: http://localhost:8171/v1"

# Keep the script running to maintain the servers
echo "All servers are running. Press Ctrl+C to stop them."
trap 'echo "Stopping servers..."; kill $server1_pid $server2_pid $server3_pid $server4_pid 2>/dev/null || true; exit 0' INT TERM

# Wait for any server to finish
wait 