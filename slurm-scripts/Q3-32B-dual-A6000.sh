#!/bin/bash
#SBATCH --job-name=Q3-32B-dual-A6000
#SBATCH --output=slurm/%j.log
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --gres=gpu:A6000:2
#SBATCH --time=48:00:00
#SBATCH --nodelist=gan.ist.berkeley.edu

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

# Clear torch compile cache to prevent hanging
export TORCH_COMPILE_CACHE_DIR=/nas/ucb/$USER/cache/torch_compile
rm -rf "$TORCH_COMPILE_CACHE_DIR" 2>/dev/null || true
mkdir -p "$TORCH_COMPILE_CACHE_DIR"

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
    local max_attempts=60  # Increased timeout for large model
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

        echo "Attempt $attempt/$max_attempts: Server not ready yet, waiting 15 seconds..."
        sleep 15
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

# ======= Launch vLLM server with tensor parallelism =======
echo "Available GPUs: $(nvidia-smi -L)"
echo "GPU 0 initial status: $(nvidia-smi -i 0 --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits)"
echo "GPU 1 initial status: $(nvidia-smi -i 1 --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits)"

# Clear GPU memory before starting
echo "Clearing GPU memory..."
nvidia-smi --gpu-reset 2>/dev/null || true
sleep 5

# Launch vLLM server with tensor parallelism across 2 GPUs
echo "Starting Qwen3-32B model with tensor parallelism across 2 A6000 GPUs on port 14320..."

# Use CUDA_VISIBLE_DEVICES to specify both GPUs
export CUDA_VISIBLE_DEVICES=0,1

# Launch with tensor parallelism
vllm serve "$MODEL_ID" \
       --host 0.0.0.0 \
       --port 14320 \
       --trust-remote-code \
       --tensor-parallel-size 2 \
       --gpu-memory-utilization 0.95 \
       --max-model-len 8192 \
       --disable-log-requests \
       --disable-log-stats \
       --dtype float16 \
       --enforce-eager > "/nas/ucb/$USER/dev/Collab-Overcooked/slurm-scripts/slurm/vllm_32B_dual.log" 2>&1 &
server_pid=$!

echo "Launched vLLM server with PID: $server_pid"
echo "GPU 0 status after launch: $(nvidia-smi -i 0 --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits)"
echo "GPU 1 status after launch: $(nvidia-smi -i 1 --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits)"

# Wait for server to be ready
if ! check_server_ready "14320" "$MODEL_ID"; then
    echo "Failed to start server on port 14320. Checking logs..."
    echo "=== Last 50 lines of vLLM log ==="
    tail -50 "/nas/ucb/$USER/dev/Collab-Overcooked/slurm-scripts/slurm/vllm_32B_dual.log" || true
    echo "=== End of log ==="
    kill $server_pid 2>/dev/null || true
    exit 1
fi

echo "Server is running and ready!"
echo "Qwen3-32B Server: http://localhost:14320/v1"
echo "Model: $MODEL_ID"
echo "Tensor Parallel Size: 2"
echo "Final GPU 0 status: $(nvidia-smi -i 0 --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits)"
echo "Final GPU 1 status: $(nvidia-smi -i 1 --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits)"

# Keep the script running to maintain the server
echo "Server is running. Press Ctrl+C to stop it."
trap 'echo "Stopping server..."; kill $server_pid 2>/dev/null || true; exit 0' INT TERM

# Wait for server to finish
wait
