#!/bin/bash
#SBATCH --job-name=Gemma-3-4B-12B-gail
#SBATCH --output=slurm/%j.log
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=48:00:00
#SBATCH --nodelist=ddpg.ist.berkeley.edu

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

# 6) Login and download all models
huggingface-cli login --token "$HF_TOKEN"
huggingface-cli download google/gemma-3-4b-it
huggingface-cli download google/gemma-3-12b-it

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

echo "Available GPUs: $(nvidia-smi -L)"
echo "GPU 0 initial status: $(nvidia-smi -i 0 --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits)"

# 7) Launch Gemma-3-4B server on port 4040 (GPU 0)
echo "Starting Gemma-3-4B model on port 4040 (GPU 0)..."
CUDA_VISIBLE_DEVICES=0 vllm serve google/gemma-3-4b-it \
       --host 0.0.0.0 \
       --port 4040 \
       --trust-remote-code \
       --gpu-memory-utilization 0.4 \
       --max-model-len 8192 > "/nas/ucb/marimeireles/dev/Collab-Overcooked/slurm-scripts/slurm/vllm_4040.log" 2>&1 &
server1_pid=$!

# Wait for first server to be ready
if ! check_server_ready "4040" "google/gemma-3-4b-it"; then
    echo "Failed to start server on port 4040. Exiting."
    kill $server1_pid 2>/dev/null || true
    exit 1
fi

# 8) Launch Gemma-3-12B server on port 4120 (GPU 0)
echo "Starting Gemma-3-12B model on port 4120 (GPU 0)..."
echo "GPU 0 status: $(nvidia-smi -i 0 --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits)"
CUDA_VISIBLE_DEVICES=0 vllm serve google/gemma-3-12b-it \
       --host 0.0.0.0 \
       --port 4120 \
       --trust-remote-code \
       --gpu-memory-utilization 0.5 \
       --max-model-len 8192 > "/nas/ucb/marimeireles/dev/Collab-Overcooked/slurm-scripts/slurm/vllm_4120.log" 2>&1 &
server2_pid=$!

# Wait for second server to be ready
if ! check_server_ready "4120" "google/gemma-3-12b-it"; then
    echo "Failed to start server on port 4120. Exiting."
    kill $server1_pid 2>/dev/null || true
    kill $server2_pid 2>/dev/null || true
    exit 1
fi

echo "Both servers are running and ready!"
echo "Gemma-3-4B Server: http://localhost:4040/v1"
echo "Gemma-3-12B Server: http://localhost:4120/v1"
echo "Final GPU 0 status: $(nvidia-smi -i 0 --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits)"

# Keep the script running to maintain the servers
echo "All servers are running. Press Ctrl+C to stop them."
trap 'echo "Stopping servers..."; kill $server1_pid $server2_pid 2>/dev/null || true; exit 0' INT TERM

# Wait for any server to finish
wait 
