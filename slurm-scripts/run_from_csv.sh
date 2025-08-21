#!/bin/bash
#SBATCH --job-name=final_for_real_
#SBATCH --output=slurm/%j.log
#SBATCH --cpus-per-task=1
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
    # Qwen3 models
    [QWEN3_32B]=4320    # Qwen3 32B
    [QWEN3_14B]=4140    # Qwen3 14B
    [QWEN3_8B]=4080     # Qwen3 8B
    [QWEN3_0_6B]=4006   # Qwen3 0.6B
    [QWEN3_1_7B]=4017   # Qwen3 1.7B
    [QWEN3_4B]=4041     # Qwen3 4B (changed from 4040 to avoid conflict)
    
    # Gemma models
    [GEMMA3_4B]=4040    # Gemma-3-4B
    [GEMMA3_12B]=4120   # Gemma-3-12B
    
    # Llama models
    [LLAMA3_1_8B]=12080 # Llama-3.1-8B
    
    # Other models
    [ACEREASON_14B]=14080 # AceReason-Nemotron-14B
    
    # Legacy/commented mappings for reference
    # [QWEN3_32B]=8320    # Qwen3 32B
    # [QWEN3_14B]=8140    # Qwen3 14B
    # [QWEN3_8B]=8080     # Qwen3 8B
    # [QWEN3_0_6B]=8006   # Qwen3 0.6B
    # [QWEN3_1_7B]=8017   # Qwen3 1.7B
    # [QWEN3_4B]=8040     # Qwen3 4B
)

model_url () { printf 'http://127.0.0.1:%s' "${SERVER_PORTS[$1]}"; }

###############################################################################
# Model directories
###############################################################################
# Qwen3 model directories
dir_qwen3_32b="/nas/ucb/marimeireles/cache/hub/models--Qwen--Qwen3-32B"
dir_qwen3_14b="/nas/ucb/marimeireles/cache/hub/models--Qwen--Qwen3-14B"
dir_qwen3_8b="/nas/ucb/marimeireles/cache/hub/models--Qwen--Qwen3-8B"
dir_qwen3_0_6b="/nas/ucb/marimeireles/cache/hub/models--Qwen--Qwen3-0.6B"
dir_qwen3_1_7b="/nas/ucb/marimeireles/cache/hub/models--Qwen--Qwen3-1.7B"
dir_qwen3_4b="/nas/ucb/marimeireles/cache/hub/models--Qwen--Qwen3-4B"

# Gemma model directories
dir_gemma3_4b="/nas/ucb/marimeireles/cache/hub/models--google--gemma-3-4b-it"
dir_gemma3_12b="/nas/ucb/marimeireles/cache/hub/models--google--gemma-3-12b-it"

# Llama model directories
dir_llama3_1_8b="/nas/ucb/marimeireles/cache/hub/models--meta-llama--Meta-Llama-3.1-8B"

# Other model directories
dir_acereason_14b="/nas/ucb/marimeireles/cache/hub/models--nvidia--AceReason-Nemotron-14B"

###############################################################################
# Convert model identifier to token and GPT format
###############################################################################
convert_model_to_formats() {
  local model_id="$1"  # e.g., "Qwen_Qwen3-14B" or "meta-llama_Meta-Llama-3.1-8B"
  
  local token gpt
  case "$model_id" in
    # Qwen models
    Qwen_Qwen3-*)
      local version=${model_id#Qwen_Qwen3-}
      token="QWEN3_${version//./_}"
      gpt="Qwen/Qwen3-${version}"
      ;;
    
    # Gemma models  
    google_gemma-3-*-it)
      local version=${model_id#google_gemma-3-}
      version=${version%-it}
      case "$version" in
        4b) token="GEMMA3_4B"; gpt="google/gemma-3-4b-it";;
        12b) token="GEMMA3_12B"; gpt="google/gemma-3-12b-it";;
        *) echo "ERROR: Unsupported Gemma version: $version" >&2; return 1;;
      esac
      ;;
    
    # Llama models
    meta-llama_Meta-Llama-3.1-8B)
      token="LLAMA3_1_8B"
      gpt="meta-llama/Meta-Llama-3.1-8B"
      ;;
    
    # Other models
    nvidia_AceReason-Nemotron-14B)
      token="ACEREASON_14B"
      gpt="nvidia/AceReason-Nemotron-14B"
      ;;
    
    *) 
      echo "ERROR: Unsupported model identifier: $model_id" >&2
      return 1
      ;;
  esac
  
  # Return values via global variables
  model_token="$token"
  model_gpt="$gpt"
}

###############################################################################
# Configure model side (p0 or p1) from a token like QWEN3_14B
###############################################################################
configure_side() {
  local side="$1"      # p0 or p1
  local token="$2"     # QWEN3_14B, QWEN3_0_6B, ...
  local gpt="$3"       # Qwen/Qwen3-14B, ...

  local server dir
  case "$token" in
    # Qwen3 models
    QWEN3_32B) server="$(model_url QWEN3_32B)/v1"; dir="$dir_qwen3_32b";;
    QWEN3_14B) server="$(model_url QWEN3_14B)/v1"; dir="$dir_qwen3_14b";;
    QWEN3_8B)  server="$(model_url QWEN3_8B)/v1";  dir="$dir_qwen3_8b";;
    QWEN3_0_6B) server="$(model_url QWEN3_0_6B)/v1"; dir="$dir_qwen3_0_6b";;
    QWEN3_1_7B) server="$(model_url QWEN3_1_7B)/v1"; dir="$dir_qwen3_1_7b";;
    QWEN3_4B)  server="$(model_url QWEN3_4B)/v1";  dir="$dir_qwen3_4b";;
    
    # Gemma models
    GEMMA3_4B) server="$(model_url GEMMA3_4B)/v1"; dir="$dir_gemma3_4b";;
    GEMMA3_12B) server="$(model_url GEMMA3_12B)/v1"; dir="$dir_gemma3_12b";;
    
    # Llama models
    LLAMA3_1_8B) server="$(model_url LLAMA3_1_8B)/v1"; dir="$dir_llama3_1_8b";;
    
    # Other models
    ACEREASON_14B) server="$(model_url ACEREASON_14B)/v1"; dir="$dir_acereason_14b";;
    
    *) echo "ERROR: Unknown model token: $token" >&2; return 1;;
  esac

  # Export side-scoped variables
  if [[ "$side" == "p0" ]]; then
    p0_server="$server"; p0_dir="$dir"; p0_gpt_model="$gpt"
  else
    p1_server="$server"; p1_dir="$dir"; p1_gpt_model="$gpt"
  fi
}

###############################################################################
# CSV input (model_pair,recipe,runs_needed)
###############################################################################
# Resolve CSV from either a provided file path, a numeric index, env vars, or array ID.
# No default fallback index; require explicit selection (loop or array).

CSV_PATH=""
CSV_INDEX_FROM_ARG=""

if [[ $# -ge 1 ]]; then
  if [[ -f "$1" ]]; then
    CSV_PATH="$1"
  elif [[ "$1" =~ ^[0-9]+$ ]]; then
    CSV_INDEX_FROM_ARG="$1"
  else
    echo "ERROR: Invalid first argument '$1'. Provide a CSV file path or a numeric index." >&2
    exit 1
  fi
fi

# Env/array provided index takes effect if no explicit file path was given
if [[ -z "$CSV_PATH" ]]; then
  CSV_INDEX="${CSV_INDEX_FROM_ARG:-${CSV_INDEX:-${ITER:-${SLURM_ARRAY_TASK_ID:-}}}}"
  if [[ -z "${CSV_INDEX}" ]]; then
    echo "ERROR: No CSV specified. Pass a CSV path or set an index via argument/env/array (e.g., 13)." >&2
    exit 1
  fi
  CSV_PATH="/nas/ucb/marimeireles/dev/Collab-Overcooked/final_for_real_${CSV_INDEX}.csv"
fi
if [[ ! -f "$CSV_PATH" ]]; then
  echo "ERROR: CSV not found at $CSV_PATH" >&2
  exit 1
fi

echo "Using CSV: $CSV_PATH"

# Skip header line; process each row
tail -n +2 "$CSV_PATH" | while IFS=, read -r model_pair recipe runs_needed; do
  # Sanitize fields: strip CR (Windows), quotes, and trim whitespace
  model_pair=${model_pair//$'\r'/}
  recipe=${recipe//$'\r'/}
  runs_needed=${runs_needed//$'\r'/}
  model_pair=${model_pair//\"/}
  recipe=${recipe//\"/}
  runs_needed=${runs_needed//\"/}
  # Trim leading/trailing whitespace
  model_pair=$(sed -E 's/^\s+//; s/\s+$//' <<< "$model_pair")
  recipe=$(sed -E 's/^\s+//; s/\s+$//' <<< "$recipe")
  runs_needed=$(sed -E 's/^\s+//; s/\s+$//' <<< "$runs_needed")

  # Normalize recipe to match prompt filenames: drop leading numeric prefix like 1_/3_/5_
  recipe_clean=$(sed -E 's/^[0-9]+[_-]//' <<< "$recipe")

  # Basic validation
  [[ -z "${model_pair:-}" || -z "${recipe:-}" || -z "${runs_needed:-}" ]] && continue
  if ! [[ "$runs_needed" =~ ^[0-9]+$ ]]; then
    echo "WARNING: Non-integer runs_needed='$runs_needed' for '$model_pair,$recipe' — skipping" >&2
    continue
  fi

  # Parse model pair - split on the longest underscore sequence to handle mixed model types
  # model_pair examples: 
  #   Qwen_Qwen3-0.6B_Qwen_Qwen3-14B
  #   Qwen_Qwen3-32B_meta-llama_Meta-Llama-3.1-8B
  #   google_gemma-3-4b-it_nvidia_AceReason-Nemotron-14B
  
  # Find the split point by identifying known model prefixes
  # We need to be smart about splitting since model names contain underscores
  p0_model=""
  p1_model=""
  
  # Try different known model patterns to find the split
  if [[ "$model_pair" =~ ^(Qwen_Qwen3-[^_]+)_(.+)$ ]]; then
    p0_model="${BASH_REMATCH[1]}"
    p1_model="${BASH_REMATCH[2]}"
  elif [[ "$model_pair" =~ ^(google_gemma-3-[^_]+)_(.+)$ ]]; then
    p0_model="${BASH_REMATCH[1]}"
    p1_model="${BASH_REMATCH[2]}"
  elif [[ "$model_pair" =~ ^(meta-llama_Meta-Llama-[^_]+)_(.+)$ ]]; then
    p0_model="${BASH_REMATCH[1]}"
    p1_model="${BASH_REMATCH[2]}"
  elif [[ "$model_pair" =~ ^(nvidia_AceReason-Nemotron-[^_]+)_(.+)$ ]]; then
    p0_model="${BASH_REMATCH[1]}"
    p1_model="${BASH_REMATCH[2]}"
  else
    echo "ERROR: Cannot parse model pair: $model_pair" >&2
    continue
  fi
  
  # Convert each model to token and GPT format
  if ! convert_model_to_formats "$p0_model"; then
    echo "ERROR: Failed to convert p0 model: $p0_model" >&2
    continue
  fi
  p0_token="$model_token"
  p0_gpt="$model_gpt"
  
  if ! convert_model_to_formats "$p1_model"; then
    echo "ERROR: Failed to convert p1 model: $p1_model" >&2
    continue
  fi
  p1_token="$model_token"
  p1_gpt="$model_gpt"

  # Configure both sides
  configure_side p0 "$p0_token" "$p0_gpt"
  configure_side p1 "$p1_token" "$p1_gpt"

  echo "=== Plan: ${p0_token} vs ${p1_token} | recipe=${recipe_clean} | runs=${runs_needed} ==="

  # Skip if either server is not reachable
  p0_root="${p0_server%/v1}"
  p1_root="${p1_server%/v1}"
  if ! server_is_ready "$p0_root"; then
      echo "WARNING: P0 server not reachable: $p0_root — skipping ${p0_token} vs ${p1_token} for ${recipe}"
      continue
  fi
  if ! server_is_ready "$p1_root"; then
      echo "WARNING: P1 server not reachable: $p1_root — skipping ${p0_token} vs ${p1_token} for ${recipe}"
      continue
  fi

  # Run the required number of times
  for i in $(seq 1 "$runs_needed"); do
    echo "— Run $i/$runs_needed — Recipe: ${recipe_clean} — ${p0_token} vs ${p1_token}"
    if srun --nodes=1 --ntasks=1 --input=none python main.py \
            --order "${recipe_clean}" \
            --temperature 0.7 \
            --file_prefix "final_for_real_" \
            --p0_gpt_model "$p0_gpt_model" \
            --p1_gpt_model "$p1_gpt_model" \
            --p0_model_dirname "$p0_dir" \
            --p1_model_dirname "$p1_dir" \
            --p0_local_server_api "$p0_server" \
            --p1_local_server_api "$p1_server"; then
        echo "Run $i completed successfully."
    else
        exit_code=$?
        echo "WARNING: Run $i failed (code $exit_code). Proceeding to next run."
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

echo "=== All CSV-driven experiments completed! ==="


