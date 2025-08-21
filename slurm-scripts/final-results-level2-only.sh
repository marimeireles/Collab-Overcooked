#!/bin/bash
#SBATCH --job-name=per_recipe
#SBATCH --output=slurm/%j.log
#SBATCH --cpus-per-task=1
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
# Environment bootstrap (same as other scripts)
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
    [QWEN3_14B]=4140    # Qwen3 14B
    [QWEN3_8B]=4080     # Qwen3 8B
    [QWEN3_0_6B]=4006   # Qwen3 0.6B
    [QWEN3_1_7B]=4017   # Qwen3 1.7B
    [QWEN3_4B]=4040     # Qwen3 4B
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
# Configure model side (p0 or p1) from a token like QWEN3_14B
###############################################################################
configure_side() {
  local side="$1"      # p0 or p1
  local token="$2"     # QWEN3_14B, QWEN3_0_6B, ...
  local gpt="$3"       # Qwen/Qwen3-14B, ...

  local server dir
  case "$token" in
    QWEN3_32B) server="$(model_url QWEN3_32B)/v1"; dir="$dir_qwen3_32b";;
    QWEN3_14B) server="$(model_url QWEN3_14B)/v1"; dir="$dir_qwen3_14b";;
    QWEN3_8B)  server="$(model_url QWEN3_8B)/v1";  dir="$dir_qwen3_8b";;
    QWEN3_0_6B) server="$(model_url QWEN3_0_6B)/v1"; dir="$dir_qwen3_0_6b";;
    QWEN3_1_7B) server="$(model_url QWEN3_1_7B)/v1"; dir="$dir_qwen3_1_7b";;
    QWEN3_4B)  server="$(model_url QWEN3_4B)/v1";  dir="$dir_qwen3_4b";;
    *) echo "ERROR: Unknown model token: $token" >&2; return 1;;
  esac

  if [[ "$side" == "p0" ]]; then
    p0_server="$server"; p0_dir="$dir"; p0_gpt_model="$gpt"
  else
    p1_server="$server"; p1_dir="$dir"; p1_gpt_model="$gpt"
  fi
}

###############################################################################
# Recipe argument (one sbatch per recipe) — accepts ANY level prefix [1-9]_
# PROMPT_DIR can be set from the environment; fallback to src/prompts
###############################################################################
PROMPT_DIR="${PROMPT_DIR:-/nas/ucb/$USER/dev/Collab-Overcooked/src/prompts}"

# Determine where the files actually live:
#   - If PROMPT_DIR itself contains files like "1_*.txt", use it directly.
#   - Else, if PROMPT_DIR/recipe exists, use that.
if compgen -G "${PROMPT_DIR}/[1-9]_*.?*" > /dev/null; then
  recipe_dir="${PROMPT_DIR}"
elif [[ -d "${PROMPT_DIR}/recipe" ]]; then
  recipe_dir="${PROMPT_DIR}/recipe"
else
  echo "ERROR: Could not locate recipe files. Checked:
  - ${PROMPT_DIR}
  - ${PROMPT_DIR}/recipe" >&2
  exit 1
fi

if [[ $# -lt 1 ]]; then
  echo "ERROR: No recipe specified." >&2
  echo "Pass the recipe *base name* (without level prefix or extension), e.g.:" >&2
  echo "  sbatch $(basename "$0") boiled_potato_slices" >&2
  exit 1
fi

recipe_clean="$1"  # e.g., "boiled_potato_slices" (works for 1_, 2_, 3_, ...)

# Accept any level prefix [1-9]_recipe_clean.*
if ! compgen -G "${recipe_dir}/[1-9]_${recipe_clean}.*" > /dev/null; then
  echo "ERROR: Recipe file not found for '${recipe_clean}' in ${recipe_dir} (expecting [1-9]_$(printf '%q' "$recipe_clean").*)" >&2
  exit 1
fi

echo "Selected recipe: ${recipe_clean}"
echo "Recipe directory: ${recipe_dir}"

###############################################################################
# Model pair selection (all ordered pairs by default; override via env)
###############################################################################
# Base model IDs (HuggingFace-style names expected by the app)
MODELS=(
  "Qwen_Qwen3-32B"
  "Qwen_Qwen3-14B"
  "Qwen_Qwen3-8B"
  "Qwen_Qwen3-4B"
  "Qwen_Qwen3-1.7B"
  "Qwen_Qwen3-0.6B"
)

# Build all ordered pairs if PAIRS is not provided.
MODEL_PAIRS=()
if [[ -n "${PAIRS:-}" ]]; then
  # Expect space-separated entries like: Qwen_Qwen3-0.6B_Qwen_Qwen3-14B
  read -r -a MODEL_PAIRS <<< "$PAIRS"
else
  for a in "${MODELS[@]}"; do
    for b in "${MODELS[@]}"; do
      MODEL_PAIRS+=("${a}_${b}")
    done
  done
fi

# Runs per model pair (override via env)
RUNS_PER_PAIR="${RUNS_PER_PAIR:-1}"
if ! [[ "$RUNS_PER_PAIR" =~ ^[0-9]+$ ]] ; then
  echo "ERROR: RUNS_PER_PAIR must be an integer" >&2; exit 1
fi

TEMPERATURE="${TEMPERATURE:-0.7}"
FILE_PREFIX="${FILE_PREFIX:-dir}"
SRUN_TIMEOUT="${SRUN_TIMEOUT:-0}"   # 0 = no timeout; else seconds (GNU timeout)

###############################################################################
# Main loops: model pairs × runs for the single requested recipe
###############################################################################
for model_pair in "${MODEL_PAIRS[@]}"; do
  # Parse huggingface-style halves from "Qwen_Qwen3-0.6B_Qwen_Qwen3-14B"
  p0_hf=$(sed -E 's/^(Qwen_Qwen3-[^_]+)_.*/\1/' <<< "$model_pair")
  p1_hf=$(sed -E 's/^.*_(Qwen_Qwen3-[^_]+)$/\1/' <<< "$model_pair")

  # Uppercase tokens for server+dir selection (QWEN3_0_6B, ...)
  p0_token=${p0_hf/Qwen_Qwen3-/QWEN3_}; p0_token=${p0_token//./_}
  p1_token=${p1_hf/Qwen_Qwen3-/QWEN3_}; p1_token=${p1_token//./_}

  # Huggingface IDs for app (Qwen/Qwen3-0.6B)
  p0_gpt=${p0_hf/_//}
  p1_gpt=${p1_hf/_//}

  # Configure both sides
  configure_side p0 "$p0_token" "$p0_gpt"
  configure_side p1 "$p1_token" "$p1_gpt"

  # Skip if either server is not reachable
  p0_root="${p0_server%/v1}"
  p1_root="${p1_server%/v1}"
  if ! server_is_ready "$p0_root"; then
      echo "WARNING: P0 server not reachable: $p0_root — skipping pair ${p0_token} vs ${p1_token}"
      continue
  fi
  if ! server_is_ready "$p1_root"; then
      echo "WARNING: P1 server not reachable: $p1_root — skipping pair ${p0_token} vs ${p1_token}"
      continue
  fi

  echo "=== Plan: ${p0_token} vs ${p1_token} | recipe=${recipe_clean} | runs=${RUNS_PER_PAIR} ==="

  for i in $(seq 1 "$RUNS_PER_PAIR"); do
    echo "— Run $i/$RUNS_PER_PAIR — ${recipe_clean} — ${p0_token} vs ${p1_token}"

    run_cmd=( srun --nodes=1 --ntasks=1 --input=none python main.py
      --order "${recipe_clean}"
      --temperature "${TEMPERATURE}"
      --file_prefix "${FILE_PREFIX}"
      --p0_gpt_model "$p0_gpt_model"
      --p1_gpt_model "$p1_gpt_model"
      --p0_model_dirname "$p0_dir"
      --p1_model_dirname "$p1_dir"
      --p0_local_server_api "$p0_server"
      --p1_local_server_api "$p1_server"
    )

    if (( SRUN_TIMEOUT > 0 )); then
      if timeout "$SRUN_TIMEOUT" "${run_cmd[@]}"; then
        echo "Run $i completed successfully."
      else
        exit_code=$?
        if [[ $exit_code == 124 ]]; then
          echo "WARNING: Run $i timed out after ${SRUN_TIMEOUT}s."
        else
          echo "WARNING: Run $i failed (code $exit_code)."
        fi
      fi
    else
      if "${run_cmd[@]}"; then
        echo "Run $i completed successfully."
      else
        exit_code=$?
        echo "WARNING: Run $i failed (code $exit_code). Proceeding to next run."
      fi
    fi

    # Flush caches between runs
    if [[ "$p0_root" == "$p1_root" ]]; then
      flush_cache "$p0_root"
    else
      flush_cache "$p0_root"; flush_cache "$p1_root"
    fi
  done

  echo "=== Completed: ${p0_token} vs ${p1_token} | recipe=${recipe_clean} ==="
done

echo "=== All experiments for recipe '${recipe_clean}' completed! ==="
