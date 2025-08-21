#!/bin/bash
#SBATCH --job-name=matrix_play
#SBATCH --output=slurm/%x_%j.log
#SBATCH --cpus-per-task=16
#SBATCH --mem=48GB
#SBATCH --time=6:00:00
#SBATCH --nodelist=ddpg.ist.berkeley.edu
set -euo pipefail

###############################################################################
# Helper: flush vLLM KV-cache for a given server root (expects no /v1 suffix)
###############################################################################
flush_cache () {
    local root="$1"
    if ! curl -s -X POST "${root}/reset_prefix_cache" > /dev/null; then
        echo "WARNING: cache flush failed for ${root}" >&2
    else
        echo "KV-cache flushed on ${root}"
    fi
}

###############################################################################
# Environment bootstrap (unchanged)
###############################################################################
set -a
source /nas/ucb/marimeireles/dev/Collab-Overcooked/slurm-scripts/secrets.env
set +a

echo "Running on host: $(hostname)"

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

###############################################################################
# External Server Configuration  (ONE source of truth)
###############################################################################
declare -A SERVER_PORTS=(
    [7B_A]=8070     # p0   Qwen 7B
    [7B_B]=8071     # p1   Qwen 7B
    [14B_A]=8140    # p0   Qwen 14B
    [14B_B]=8141    # p1   Qwen 14B
    [8B_LLAMA_A]=4140  # p0   Llama 8B
    [8B_LLAMA_B]=4141  # p1   Llama 8B
)

model_url () { printf 'http://0.0.0.0:%s' "${SERVER_PORTS[$1]}"; }

###############################################################################
# Model directories
###############################################################################
dir_14b="/nas/ucb/marimeireles/cache/hub/models--Qwen--Qwen2.5-14B-Instruct"
dir_7b="/nas/ucb/marimeireles/cache/hub/models--Qwen--Qwen2.5-7B-Instruct"
dir_8b_llama="/nas/ucb/marimeireles/cache/hub/models--meta-llama--Meta-Llama-3-8B-Instruct"

###############################################################################
# Model-combination matrix (all permutations)
###############################################################################
combinations=(
    "14B 7B"
    "14B 14B"
    "14B 8B_LLAMA"
    "7B 14B"
    "7B 7B"
    "7B 8B_LLAMA"
    "8B_LLAMA 14B"
    "8B_LLAMA 7B"
    "8B_LLAMA 8B_LLAMA"
)

echo "=== Running model combination matrix (1 iteration) ==="

for combo in "${combinations[@]}"; do
    read -r p0_model p1_model <<< "$combo"

    # Select endpoint and local model directory for p0
    if [[ $p0_model == 14B ]]; then
        p0_server="$(model_url 14B_A)/v1"
        p0_dir="$dir_14b"
        p0_gpt_model="Qwen/Qwen2.5-14B-Instruct"
    elif [[ $p0_model == 7B ]]; then
        p0_server="$(model_url 7B_A)/v1"
        p0_dir="$dir_7b"
        p0_gpt_model="Qwen/Qwen2.5-7B-Instruct"
    elif [[ $p0_model == 8B_LLAMA ]]; then
        p0_server="$(model_url 8B_LLAMA_A)/v1"
        p0_dir="$dir_8b_llama"
        p0_gpt_model="meta-llama/Meta-Llama-3-8B-Instruct"
    fi

    # Select endpoint and local model directory for p1
    if [[ $p1_model == 14B ]]; then
        p1_server="$(model_url 14B_B)/v1"
        p1_dir="$dir_14b"
        p1_gpt_model="Qwen/Qwen2.5-14B-Instruct"
    elif [[ $p1_model == 7B ]]; then
        p1_server="$(model_url 7B_B)/v1"
        p1_dir="$dir_7b"
        p1_gpt_model="Qwen/Qwen2.5-7B-Instruct"
    elif [[ $p1_model == 8B_LLAMA ]]; then
        p1_server="$(model_url 8B_LLAMA_B)/v1"
        p1_dir="$dir_8b_llama"
        p1_gpt_model="meta-llama/Meta-Llama-3-8B-Instruct"
    fi

    echo "=== Starting experiment: ${p0_model} (p0) vs ${p1_model} (p1) ==="
    echo "P0 Server: $p0_server"
    echo "P1 Server: $p1_server"

    # ---------------------------------------------------------------------
    # Run the experiment (60-min timeout) .................................
    # ---------------------------------------------------------------------
    if timeout 3600 srun --nodes=1 --ntasks=1 python main.py \
            --order boiled_egg \
            --p0_gpt_model "$p0_gpt_model" \
            --p1_gpt_model "$p1_gpt_model" \
            --p0_model_dirname "$p0_dir" \
            --p1_model_dirname "$p1_dir" \
            --p0_local_server_api "$p0_server" \
            --p1_local_server_api "$p1_server"; then
        echo "Experiment ${p0_model} vs ${p1_model} completed successfully."
    else
        exit_code=$?
        if [[ $exit_code == 124 ]]; then
            echo "WARNING: Experiment ${p0_model} vs ${p1_model} timed out."
        else
            echo "WARNING: Experiment ${p0_model} vs ${p1_model} failed (code $exit_code)."
        fi
    fi

    # ---------------------------------------------------------------------
    # Flush KV-cache on both engines .......................................
    # ---------------------------------------------------------------------
    flush_cache "${p0_server%/v1}"
    flush_cache "${p1_server%/v1}"

    echo "=== Completed experiment: ${p0_model} vs ${p1_model} ==="
done

echo "=== All experiments completed! ==="
