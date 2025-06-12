#!/bin/bash
#SBATCH --job-name=vllm-serve
#SBATCH --output=slurm/%j.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:A100-SXM4-80GB:1
#SBATCH --time=12:00:00

set -euo pipefail

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
export HUGGINGFACE_HUB_TOKEN=""
huggingface-cli login --token "$HUGGINGFACE_HUB_TOKEN"
huggingface-cli download Qwen/Qwen2.5-7B-Instruct

# 7) Launch vLLM server
srun --nodes=1 --ntasks=1 \
     vllm serve Qwen/Qwen2.5-7B-Instruct \
       --host 0.0.0.0 \
       --port 8000
