#!/bin/bash
#SBATCH --job-name=Q3-1-3
#SBATCH --output=slurm/%j.log
#SBATCH --cpus-per-task=8
#SBATCH --mem=8GB
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
# Helper: check if local server is reachable (no model assumption)
###############################################################################
server_is_ready () {
    local root="$1"   # expects no /v1 suffix
    curl -s -m 3 "${root}/v1/models" > /dev/null
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
# External Server Configuration  (ONE server per LLM model)
###############################################################################
declare -A SERVER_PORTS=(
    [QWEN3_32B]=8320    # Qwen3 32B
    [QWEN3_14B]=8140    # Qwen3 14B
    [QWEN3_8B]=8080     # Qwen3 8B
    [QWEN3_0_6B]=8006   # Qwen3 0.6B
    [QWEN3_1_7B]=8017   # Qwen3 1.7B
    [QWEN3_4B]=8040     # Qwen3 4B
)

model_url () { printf 'http://127.0.0.1:%s' "${SERVER_PORTS[$1]}"; }

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
# Model-combination matrix (all permutations)
###############################################################################
combinations=(
    'QWEN3_32B QWEN3_14B'
    'QWEN3_8B QWEN3_32B'
    # 'QWEN3_0_6B QWEN3_0_6B'
    'QWEN3_8B QWEN3_8B'
    # 'QWEN3_1_7B QWEN3_4B'
    'QWEN3_32B QWEN3_8B'
    'QWEN3_14B QWEN3_1_7B'
    'QWEN3_4B QWEN3_14B'
    'QWEN3_1_7B QWEN3_8B'
    'QWEN3_14B QWEN3_4B'
    'QWEN3_14B QWEN3_14B'
    'QWEN3_4B QWEN3_4B'
    'QWEN3_32B QWEN3_1_7B'
    'QWEN3_14B QWEN3_8B'
    'QWEN3_32B QWEN3_4B'
    # 'QWEN3_1_7B QWEN3_1_7B'
    # 'QWEN3_4B QWEN3_1_7B'
    'QWEN3_8B QWEN3_1_7B'
    'QWEN3_32B QWEN3_32B'
    'QWEN3_0_6B QWEN3_8B'
    'QWEN3_0_6B QWEN3_14B'
    'QWEN3_0_6B QWEN3_32B'
    # 'QWEN3_0_6B QWEN3_1_7B'
    # 'QWEN3_0_6B QWEN3_4B'
    'QWEN3_1_7B QWEN3_32B'
    'QWEN3_1_7B QWEN3_14B'
    'QWEN3_4B QWEN3_32B'
    'QWEN3_4B QWEN3_8B'
    'QWEN3_8B QWEN3_14B'
    'QWEN3_8B QWEN3_4B'
    'QWEN3_14B QWEN3_32B'
    'QWEN3_14B QWEN3_0_6B'
    'QWEN3_32B QWEN3_0_6B'
)

###############################################################################
# Randomize combinations once (optionally reproducible with $COMBO_SEED)
###############################################################################
randomize_combinations() {
  # Produces: RANDOMIZED_COMBINATIONS (global array)
  if command -v shuf >/dev/null 2>&1; then
    if [[ -n "${COMBO_SEED:-}" ]] && command -v openssl >/dev/null 2>&1; then
      # Deterministic bytes source for shuf using the seed
      mapfile -t RANDOMIZED_COMBINATIONS < <(
        printf '%s\n' "${combinations[@]}" \
        | shuf --random-source=<(openssl enc -aes-256-ctr -pass pass:"$COMBO_SEED" -nosalt </dev/zero 2>/dev/null)
      )
    elif [[ -n "${COMBO_SEED:-}" ]]; then
      # Deterministic fallback using awk + sort (no openssl)
      mapfile -t RANDOMIZED_COMBINATIONS < <(
        printf '%s\n' "${combinations[@]}" \
        | awk -v seed="$COMBO_SEED" 'BEGIN{srand(seed)}{printf "%0.9f\t%s\n", rand(), $0}' \
        | sort -n | cut -f2-
      )
    else
      # Non‑deterministic shuffle
      mapfile -t RANDOMIZED_COMBINATIONS < <(printf '%s\n' "${combinations[@]}" | shuf)
    fi
  else
    # Pure Bash Fisher–Yates shuffle
    RANDOMIZED_COMBINATIONS=("${combinations[@]}")
    local n=${#RANDOMIZED_COMBINATIONS[@]}
    for ((i=n-1; i>0; i--)); do
      j=$((RANDOM % (i+1)))
      tmp=${RANDOMIZED_COMBINATIONS[i]}
      RANDOMIZED_COMBINATIONS[i]=${RANDOMIZED_COMBINATIONS[j]}
      RANDOMIZED_COMBINATIONS[j]=$tmp
    done
  fi
}

randomize_combinations

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
    
    for combo in "${RANDOMIZED_COMBINATIONS[@]}"; do
    read -r p0_model p1_model <<< "$combo"

        # Select endpoint and local model directory for p0
        if [[ $p0_model == QWEN3_32B ]]; then
            p0_server="$(model_url QWEN3_32B)/v1"
            p0_dir="$dir_qwen3_32b"
            p0_gpt_model="Qwen/Qwen3-32B"
        elif [[ $p0_model == QWEN3_14B ]]; then
            p0_server="$(model_url QWEN3_14B)/v1"
            p0_dir="$dir_qwen3_14b"
            p0_gpt_model="Qwen/Qwen3-14B"
        elif [[ $p0_model == QWEN3_8B ]]; then
            p0_server="$(model_url QWEN3_8B)/v1"
            p0_dir="$dir_qwen3_8b"
            p0_gpt_model="Qwen/Qwen3-8B"
        elif [[ $p0_model == QWEN3_0_6B ]]; then
            p0_server="$(model_url QWEN3_0_6B)/v1"
            p0_dir="$dir_qwen3_0_6b"
            p0_gpt_model="Qwen/Qwen3-0.6B"
        elif [[ $p0_model == QWEN3_1_7B ]]; then
            p0_server="$(model_url QWEN3_1_7B)/v1"
            p0_dir="$dir_qwen3_1_7b"
            p0_gpt_model="Qwen/Qwen3-1.7B"
        elif [[ $p0_model == QWEN3_4B ]]; then
            p0_server="$(model_url QWEN3_4B)/v1"
            p0_dir="$dir_qwen3_4b"
            p0_gpt_model="Qwen/Qwen3-4B"
        fi

        # Select endpoint and local model directory for p1
        if [[ $p1_model == QWEN3_32B ]]; then
            p1_server="$(model_url QWEN3_32B)/v1"
            p1_dir="$dir_qwen3_32b"
            p1_gpt_model="Qwen/Qwen3-32B"
        elif [[ $p1_model == QWEN3_14B ]]; then
            p1_server="$(model_url QWEN3_14B)/v1"
            p1_dir="$dir_qwen3_14b"
            p1_gpt_model="Qwen/Qwen3-14B"
        elif [[ $p1_model == QWEN3_8B ]]; then
            p1_server="$(model_url QWEN3_8B)/v1"
            p1_dir="$dir_qwen3_8b"
            p1_gpt_model="Qwen/Qwen3-8B"
        elif [[ $p1_model == QWEN3_0_6B ]]; then
            p1_server="$(model_url QWEN3_0_6B)/v1"
            p1_dir="$dir_qwen3_0_6b"
            p1_gpt_model="Qwen/Qwen3-0.6B"
        elif [[ $p1_model == QWEN3_1_7B ]]; then
            p1_server="$(model_url QWEN3_1_7B)/v1"
            p1_dir="$dir_qwen3_1_7b"
            p1_gpt_model="Qwen/Qwen3-1.7B"
        elif [[ $p1_model == QWEN3_4B ]]; then
            p1_server="$(model_url QWEN3_4B)/v1"
            p1_dir="$dir_qwen3_4b"
            p1_gpt_model="Qwen/Qwen3-4B"
        fi

        echo "=== Starting experiment: ${p0_model} (p0) vs ${p1_model} (p1) ==="
        echo "P0 Server: $p0_server"
        echo "P1 Server: $p1_server"

        # Skip if either server is not reachable
        p0_root="${p0_server%/v1}"
        p1_root="${p1_server%/v1}"
        if ! server_is_ready "$p0_root"; then
            echo "WARNING: P0 server not reachable: $p0_root — skipping combo ${p0_model} vs ${p1_model}"
            continue
        fi
        if ! server_is_ready "$p1_root"; then
            echo "WARNING: P1 server not reachable: $p1_root — skipping combo ${p0_model} vs ${p1_model}"
            continue
        fi

        if srun --nodes=1 --ntasks=1 python main.py \
                --order "${recipe}" \
                --temperature 0.7 \
                --file_prefix "required" \
                --p0_gpt_model "$p0_gpt_model" \
                --p1_gpt_model "$p1_gpt_model" \
                --p0_model_dirname "$p0_dir" \
                --p1_model_dirname "$p1_dir" \
                --p0_local_server_api "$p0_server" \
                --p1_local_server_api "$p1_server"; then
            echo "Experiment ${p0_model} vs ${p1_model} completed successfully."
        else
            echo "WARNING: Experiment ${p0_model} vs ${p1_model} failed (code $exit_code)."
        fi

        # ---------------------------------------------------------------------
        # Flush KV-cache on unique engines
        # ---------------------------------------------------------------------
        if [[ "${p0_server%/v1}" == "${p1_server%/v1}" ]]; then
            flush_cache "${p0_server%/v1}"
        else
            flush_cache "${p0_server%/v1}"
            flush_cache "${p1_server%/v1}"
        fi

        echo "=== Completed experiment: ${p0_model} vs ${p1_model} ==="
    done
    done    # recipe loop
done

echo "=== All experiments completed! ==="

