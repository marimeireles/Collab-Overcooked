#!/bin/bash
#SBATCH --job-name=repl_07_07_h1853
#SBATCH --output=slurm/%x_%j.log
#SBATCH --cpus-per-task=16
#SBATCH --mem=36GB
#SBATCH --time=22:00:00
#SBATCH --nodelist=sac.ist.berkeley.edu
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
    [MISTRAL_7B_A]=6140  # p0   Mistral 7B
    [MISTRAL_7B_B]=6141  # p1   Mistral 7B
    [QWEN_32B_A]=10000   # p0   Qwen 32B
    [QWEN_32B_B]=10001   # p1   Qwen 32B
)

model_url () { printf 'http://0.0.0.0:%s' "${SERVER_PORTS[$1]}"; }

###############################################################################
# Model directories
###############################################################################
dir_14b="/nas/ucb/marimeireles/cache/hub/models--Qwen--Qwen2.5-14B-Instruct"
dir_7b="/nas/ucb/marimeireles/cache/hub/models--Qwen--Qwen2.5-7B-Instruct"
dir_8b_llama="/nas/ucb/marimeireles/cache/hub/models--meta-llama--Meta-Llama-3-8B-Instruct"
dir_mistral_7b="/nas/ucb/marimeireles/cache/hub/models--mistralai--Mistral-7B-Instruct-v0.1"
dir_qwen_32b="/nas/ucb/marimeireles/models/qwen2.5-32b"

###############################################################################
# Model-combination matrix (all permutations)
###############################################################################
combinations=(
    # '7B 14B'
    # '7B 8B_LLAMA'
    # '8B_LLAMA QWEN_32B'
    # '8B_LLAMA MISTRAL_7B'
    # '8B_LLAMA 8B_LLAMA'
    # 'MISTRAL_7B QWEN_32B'
    '14B 14B'
    # 'QWEN_32B MISTRAL_7B'
    # '14B 8B_LLAMA'
    # 'QWEN_32B QWEN_32B'
    # '7B QWEN_32B'
    # '8B_LLAMA 14B'
    '7B 7B'
    # 'QWEN_32B 14B'
    # '14B 7B'
    # 'MISTRAL_7B 8B_LLAMA'
    # 'MISTRAL_7B MISTRAL_7B'
    # '14B QWEN_32B'
    # '14B MISTRAL_7B'
    # 'MISTRAL_7B 14B'
    # 'QWEN_32B 8B_LLAMA'
    # '8B_LLAMA 7B'
    # 'MISTRAL_7B 7B'
    # '7B MISTRAL_7B'
    # 'QWEN_32B 7B'
)

PROMPT_DIR="$(pwd)/prompts"
recipe_dir="${PROMPT_DIR}/recipe"

mapfile -t level1_recipes < <(
  find "${recipe_dir}" -maxdepth 1 -type f -name '1_*' \
       -printf '%f\n' |               # basename only
  sed -E 's/^1_//;s/\.[^.]+$//' |     # drop leading "1_" and the extension
  sort -u
)

if (( ${#level1_recipes[@]} == 0 )); then
  echo "ERROR: no level-1 recipes (files named 1_*) found in ${recipe_dir}" >&2
  exit 1
fi

for iteration in {1..5}; do
    echo "=== Starting iteration $iteration ==="

  for recipe in "${level1_recipes[@]}"; do
    echo "— Recipe: ${recipe} —"
    
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
        elif [[ $p0_model == MISTRAL_7B ]]; then
            p0_server="$(model_url MISTRAL_7B_A)/v1"
            p0_dir="$dir_mistral_7b"
            p0_gpt_model="mistralai/Mistral-7B-Instruct-v0.1"
        elif [[ $p0_model == QWEN_32B ]]; then
            p0_server="$(model_url QWEN_32B_A)/v1"
            p0_dir="$dir_qwen_32b"
            p0_gpt_model="Qwen/Qwen2.5-32B-Instruct"
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
        elif [[ $p1_model == MISTRAL_7B ]]; then
            p1_server="$(model_url MISTRAL_7B_B)/v1"
            p1_dir="$dir_mistral_7b"
            p1_gpt_model="mistralai/Mistral-7B-Instruct-v0.1"
        elif [[ $p1_model == QWEN_32B ]]; then
            p1_server="$(model_url QWEN_32B_B)/v1"
            p1_dir="$dir_qwen_32b"
            p1_gpt_model="Qwen/Qwen2.5-32B-Instruct"
        fi

        echo "=== Starting experiment: ${p0_model} (p0) vs ${p1_model} (p1) ==="
        echo "P0 Server: $p0_server"
        echo "P1 Server: $p1_server"

        # ---------------------------------------------------------------------
        # Run the experiment (60-min timeout) .................................
        # ---------------------------------------------------------------------
        if timeout 3600 srun --nodes=1 --ntasks=1 python main.py \
                --order "${recipe}" \
                --temperature 0.7 \
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
  done    # recipe loop
    echo "=== Completed iteration $iteration of 5 ==="
done

echo "=== All experiments completed! ==="

