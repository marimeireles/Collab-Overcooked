#!/bin/bash
#SBATCH --job-name=Q3-combo
#SBATCH --output=slurm/%j_combo.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=4GB
#SBATCH --time=48:00:00
#SBATCH --nodelist=airl.ist.berkeley.edu
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
    [QWEN3_32B_A]=8320     # p0   Qwen3 32B
    [QWEN3_32B_B]=8321     # p1   Qwen3 32B
    [QWEN3_14B_A]=8140     # p0   Qwen3 14B
    [QWEN3_14B_B]=8141     # p1   Qwen3 14B
    [QWEN3_8B_A]=8080      # p0   Qwen3 8B
    [QWEN3_8B_B]=8081      # p1   Qwen3 8B
    [QWEN3_0_6B_A]=8060    # p0   Qwen3 0.6B
    [QWEN3_0_6B_B]=8061    # p1   Qwen3 0.6B
    [QWEN3_1_7B_A]=8170    # p0   Qwen3 1.7B
    [QWEN3_1_7B_B]=8171    # p1   Qwen3 1.7B
    [QWEN3_4B_A]=8040      # p0   Qwen3 4B
    [QWEN3_4B_B]=8041      # p1   Qwen3 4B
)

model_url () { printf 'http://0.0.0.0:%s' "${SERVER_PORTS[$1]}"; }

###############################################################################
# Model directories
###############################################################################
dir_qwen3_32b="/nas/ucb/marimeireles/cache/hub/models--Qwen--Qwen3-32B"
dir_qwen3_14b="/nas/ucb/marimeireles/cache/hub/models--Qwen--Qwen3-14B"
dir_qwen3_8b="/nas/ucb/marimeireles/cache/hub/models--Qwen--Qwen3-8B"
dir_qwen3_0_6b="/nas/ucb/marimeireles/cache/hub/models--Qwen--Qwen3-0.6B"
dir_qwen3_1_7b="/nas/ucb/marimeireles/cache/hub/models--Qwen--Qwen3-1.7B"
dir_qwen3_4b="/nas/ucb/marimeireles/cache/hub/models--Qwen--Qwen3-4B"

###############################################################################
# Single combination for this job
###############################################################################
combo='QWEN3_14B QWEN3_1_7B'

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

# run the experiment 2 times
for i in {1..2}; do
    for recipe in "${level1_recipes[@]}"; do
    echo "— Recipe: ${recipe} —"
    
    read -r p0_model p1_model <<< "$combo"

        # Select endpoint and local model directory for p0
        if [[ $p0_model == QWEN3_32B ]]; then
            p0_server="$(model_url QWEN3_32B_A)/v1"
            p0_dir="$dir_qwen3_32b"
            p0_gpt_model="Qwen/Qwen3-32B"
        elif [[ $p0_model == QWEN3_14B ]]; then
            p0_server="$(model_url QWEN3_14B_A)/v1"
            p0_dir="$dir_qwen3_14b"
            p0_gpt_model="Qwen/Qwen3-14B"
        elif [[ $p0_model == QWEN3_8B ]]; then
            p0_server="$(model_url QWEN3_8B_A)/v1"
            p0_dir="$dir_qwen3_8b"
            p0_gpt_model="Qwen/Qwen3-8B"
        elif [[ $p0_model == QWEN3_0_6B ]]; then
            p0_server="$(model_url QWEN3_0_6B_A)/v1"
            p0_dir="$dir_qwen3_0_6b"
            p0_gpt_model="Qwen/Qwen3-0.6B"
        elif [[ $p0_model == QWEN3_1_7B ]]; then
            p0_server="$(model_url QWEN3_1_7B_A)/v1"
            p0_dir="$dir_qwen3_1_7b"
            p0_gpt_model="Qwen/Qwen3-1.7B"
        elif [[ $p0_model == QWEN3_4B ]]; then
            p0_server="$(model_url QWEN3_4B_A)/v1"
            p0_dir="$dir_qwen3_4b"
            p0_gpt_model="Qwen/Qwen3-4B"
        fi

        # Select endpoint and local model directory for p1
        if [[ $p1_model == QWEN3_32B ]]; then
            p1_server="$(model_url QWEN3_32B_B)/v1"
            p1_dir="$dir_qwen3_32b"
            p1_gpt_model="Qwen/Qwen3-32B"
        elif [[ $p1_model == QWEN3_14B ]]; then
            p1_server="$(model_url QWEN3_14B_B)/v1"
            p1_dir="$dir_qwen3_14b"
            p1_gpt_model="Qwen/Qwen3-14B"
        elif [[ $p1_model == QWEN3_8B ]]; then
            p1_server="$(model_url QWEN3_8B_B)/v1"
            p1_dir="$dir_qwen3_8b"
            p1_gpt_model="Qwen/Qwen3-8B"
        elif [[ $p1_model == QWEN3_0_6B ]]; then
            p1_server="$(model_url QWEN3_0_6B_B)/v1"
            p1_dir="$dir_qwen3_0_6b"
            p1_gpt_model="Qwen/Qwen3-0.6B"
        elif [[ $p1_model == QWEN3_1_7B ]]; then
            p1_server="$(model_url QWEN3_1_7B_B)/v1"
            p1_dir="$dir_qwen3_1_7b"
            p1_gpt_model="Qwen/Qwen3-1.7B"
        elif [[ $p1_model == QWEN3_4B ]]; then
            p1_server="$(model_url QWEN3_4B_B)/v1"
            p1_dir="$dir_qwen3_4b"
            p1_gpt_model="Qwen/Qwen3-4B"
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
        # Flush KV-cache on both engines
        # ---------------------------------------------------------------------
        flush_cache "${p0_server%/v1}"
        flush_cache "${p1_server%/v1}"

        echo "=== Completed experiment: ${p0_model} vs ${p1_model} ==="
    done    # recipe loop
done

echo "=== All experiments completed! ==="
