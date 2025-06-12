#!/bin/bash
#SBATCH --job-name=qwen7b_overcooked
#SBATCH --output=slurm/%j.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=4:00:00
#SBATCH --nodelist=airl.ist.berkeley.edu

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
export OPENAI_API_KEY=""

# 3) Initialize micromamba shell environment
eval "$($MAMBA_ROOT_PREFIX/micromamba shell hook --shell bash)"

# 4) Activate environment
micromamba activate $MAMBA_ROOT_PREFIX/envs/overcooked

# 5) Change to project directory
cd /nas/ucb/$USER/dev/Collab-Overcooked/src

# 6) Run Overcooked client
srun --nodes=1 --ntasks=1 python /nas/ucb/marimeireles/dev/Collab-Overcooked/src/main.py \
      --order boiled_egg \
      --p0_gpt_model "gpt-3.5-turbo-0125" \
      --p1_gpt_model "Qwen/Qwen2.5-7B-Instruct" \
      --p1_model_dirname "/nas/ucb/marimeireles/cache/hub/models--Qwen--Qwen2.5-7B-Instruct" \
      --p1_local_server_api "http://0.0.0.0:8000/v1"
