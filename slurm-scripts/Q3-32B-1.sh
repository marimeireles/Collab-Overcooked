#!/bin/bash
#SBATCH --job-name=Q3-32B-1
#SBATCH --output=slurm/%j.log
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --gres=gpu:A100-SXM4-80GB:2
#SBATCH --time=48:00:00
#SBATCH --nodelist=sac.ist.berkeley.edu

set -euo pipefail
set -a
source /nas/ucb/marimeireles/dev/Collab-Overcooked/slurm-scripts/secrets.env
set +a

echo "Running on host: $(hostname)"

# ======= Paths and environment =======
export MAMBA_ROOT_PREFIX=/nas/ucb/$USER/micromamba
export PATH=$MAMBA_ROOT_PREFIX/bin:$PATH

# Scratch & cache locations
export TMPDIR=/nas/ucb/$USER/pip_tmp
export HF_HOME=/nas/ucb/$USER/cache/hub
export HUGGINGFACE_HUB_CACHE=$HF_HOME
export XDG_CONFIG_HOME=/nas/ucb/$USER/.config
export OUTLINES_CACHE_DIR=/nas/ucb/$USER/cache/outlines

# Your personal Hugging Face token **must** be set in the job environment
: "${HF_TOKEN:?Environment variable HF_TOKEN is not set!}"

# ======= Activate software stack =======
eval "$($MAMBA_ROOT_PREFIX/micromamba shell hook --shell bash)"
micromamba activate overcooked

# Make sure the bug-fixed version of huggingface-hub is available
python -m pip install --quiet --upgrade "huggingface_hub>=0.22.2"

# ======= Project workspace =======
WORKDIR=/nas/ucb/$USER/dev/Collab-Overcooked
cd "$WORKDIR"

# Function to check if server is ready
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

# ======= Model download (if not already cached) =======
MODEL_ID="Qwen/Qwen3-32B"
MODEL_DIR=/nas/ucb/$USER/models/qwen3-32B

# Login to Hugging Face
huggingface-cli login --token "$HF_TOKEN"

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "[+] Downloading $MODEL_ID to $MODEL_DIR ..."
    huggingface-cli download "$MODEL_ID" \
        --local-dir "$MODEL_DIR" \
        --local-dir-use-symlinks False \
        --token "$HF_TOKEN"
else
    echo "[+] Using cached model in $MODEL_DIR"
fi

# ======= Launch vLLM servers =======
echo "Available GPUs: $(nvidia-smi -L)"

# Launch first vLLM server on port 8320 (GPU 0)
echo "Starting first Qwen 32B model on port 8320 (GPU 0)..."
CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL_ID" \
       --host 0.0.0.0 \
       --port 8320 \
       --trust-remote-code \
       --gpu-memory-utilization 0.90 \
       --max-model-len 26417 > "/nas/ucb/$USER/dev/Collab-Overcooked/slurm-scripts/slurm/vllm_8320.log" 2>&1 &
server1_pid=$!

# Wait for first server to be ready
if ! check_server_ready "8320" "$MODEL_ID"; then
    echo "Failed to start server on port 8320. Exiting."
    kill $server1_pid 2>/dev/null || true
    exit 1
fi

# Launch second vLLM server on port 8321 (GPU 1)
echo "Starting second Qwen 32B model on port 8321 (GPU 1)..."
echo "GPU 1 status: $(nvidia-smi -i 1 --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits)"
CUDA_VISIBLE_DEVICES=1 vllm serve "$MODEL_ID" \
       --host 0.0.0.0 \
       --port 8321 \
       --trust-remote-code \
       --gpu-memory-utilization 0.90 \
       --max-model-len 26417 > "/nas/ucb/$USER/dev/Collab-Overcooked/slurm-scripts/slurm/vllm_8321.log" 2>&1 &
server2_pid=$!

# Wait for second server to be ready
if ! check_server_ready "8321" "$MODEL_ID"; then
    echo "Failed to start server on port 8321. Exiting."
    kill $server1_pid 2>/dev/null || true
    kill $server2_pid 2>/dev/null || true
    exit 1
fi

echo "Both servers are running and ready!"
echo "Server 1: http://localhost:8320/v1"
echo "Server 2: http://localhost:8321/v1"
echo "Model: $MODEL_ID"

# Keep the script running to maintain the servers
echo "Servers are running. Press Ctrl+C to stop them."
trap 'echo "Stopping servers..."; kill $server1_pid $server2_pid 2>/dev/null || true; exit 0' INT TERM

# Wait for either server to finish
wait
