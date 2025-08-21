#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=48:00:00
#SBATCH --nodelist=airl.ist.berkeley.edu

set -euo pipefail
set -a
source /nas/ucb/marimeireles/dev/Collab-Overcooked/slurm-scripts/secrets.env
set +a

echo "Running job $JOB_ID on host: $(hostname)"

# Environment setup
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

# Server ports mapping
declare -A SERVER_PORTS=(
    [QWEN3_32B]=8320
    [QWEN3_14B]=8140
    [QWEN3_8B]=8080
    [QWEN3_0_6B]=8006
    [QWEN3_1_7B]=8017
    [QWEN3_4B]=8040
)

model_url () { printf 'http://0.0.0.0:%s' "${SERVER_PORTS[$1]}"; }

# Model directories
dir_qwen3_32b="/nas/ucb/marimeireles/cache/hub/models--Qwen--Qwen3-32B"
dir_qwen3_14b="/nas/ucb/marimeireles/cache/hub/models--Qwen--Qwen3-14B"
dir_qwen3_8b="/nas/ucb/marimeireles/cache/hub/models--Qwen--Qwen3-8B"
dir_qwen3_0_6b="/nas/ucb/marimeireles/cache/hub/models--Qwen--Qwen3-0.6B"
dir_qwen3_1_7b="/nas/ucb/marimeireles/cache/hub/models--Qwen--Qwen3-1.7B"
dir_qwen3_4b="/nas/ucb/marimeireles/cache/hub/models--Qwen--Qwen3-4B"

# KV-cache flush function
flush_cache () {
    local root="$1"
    if ! curl -s -X POST "${root}/reset_prefix_cache" > /dev/null; then
        echo "WARNING: cache flush failed for ${root}" >&2
    else
        echo "KV-cache flushed on ${root}"
    fi
}

# Get model configuration
get_model_config() {
    local model="$1"
    local prefix="$2"
    
    case $model in
        QWEN3_32B)
            eval "${prefix}_server=\"$(model_url QWEN3_32B)/v1\""
            eval "${prefix}_dir=\"$dir_qwen3_32b\""
            eval "${prefix}_gpt_model=\"Qwen/Qwen3-32B\""
            ;;
        QWEN3_14B)
            eval "${prefix}_server=\"$(model_url QWEN3_14B)/v1\""
            eval "${prefix}_dir=\"$dir_qwen3_14b\""
            eval "${prefix}_gpt_model=\"Qwen/Qwen3-14B\""
            ;;
        QWEN3_8B)
            eval "${prefix}_server=\"$(model_url QWEN3_8B)/v1\""
            eval "${prefix}_dir=\"$dir_qwen3_8b\""
            eval "${prefix}_gpt_model=\"Qwen/Qwen3-8B\""
            ;;
        QWEN3_0_6B)
            eval "${prefix}_server=\"$(model_url QWEN3_0_6B)/v1\""
            eval "${prefix}_dir=\"$dir_qwen3_0_6b\""
            eval "${prefix}_gpt_model=\"Qwen/Qwen3-0.6B\""
            ;;
        QWEN3_1_7B)
            eval "${prefix}_server=\"$(model_url QWEN3_1_7B)/v1\""
            eval "${prefix}_dir=\"$dir_qwen3_1_7b\""
            eval "${prefix}_gpt_model=\"Qwen/Qwen3-1.7B\""
            ;;
        QWEN3_4B)
            eval "${prefix}_server=\"$(model_url QWEN3_4B)/v1\""
            eval "${prefix}_dir=\"$dir_qwen3_4b\""
            eval "${prefix}_gpt_model=\"Qwen/Qwen3-4B\""
            ;;
    eairl
}

# Read experiments from file
echo "🔍 DEBUG: Looking for experiment file: $EXPERIMENT_FILE"
echo "🔍 DEBUG: File exists check: $(test -f "$EXPERIMENT_FILE" && echo "YES" || echo "NO")"
echo "🔍 DEBUG: Current directory: $(pwd)"
echo "🔍 DEBUG: Directory listing of $(dirname "$EXPERIMENT_FILE"):"
ls -la "$(dirname "$EXPERIMENT_FILE")" || echo "Directory not accessible"

if [[ ! -f "$EXPERIMENT_FILE" ]]; then
    echo "ERROR: Experiment file $EXPERIMENT_FILE not found!"
    echo "Available files in job_data directory:"
    ls -la "/nas/ucb/$USER/dev/Collab-Overcooked/slurm-scripts/job_data/" || echo "job_data directory not found"
    exit 1
fi

# Count experiments
total_experiments=$(wc -l < "$EXPERIMENT_FILE")
echo "📋 Job $JOB_ID processing $total_experiments experiments"

# Process each experiment
exp_count=0
echo "🔍 DEBUG: Starting to read experiments from file..."
echo "🔍 DEBUG: First few lines of experiment file:"
head -3 "$EXPERIMENT_FILE"

echo "🔍 DEBUG: About to start while loop..."
while IFS='|' read -r iteration recipe p0_model p1_model; do
    ((exp_count++))
    
    echo "🔍 DEBUG: INSIDE LOOP - Read line: iteration=$iteration, recipe=$recipe, p0_model=$p0_model, p1_model=$p1_model"
    echo "🎯 [$JOB_ID:$exp_count/$total_experiments] Starting: $p0_model vs $p1_model on $recipe (iter $iteration)"
    
    # Get model configurations
    get_model_config "$p0_model" "p0"
    get_model_config "$p1_model" "p1"
    
    # Create log file
    timestamp=$(date '+%Y%m%d_%H%M%S')
    safe_p0=$(echo "$p0_gpt_model" | sed 's|/|_|g; s|-|_|g')
    safe_p1=$(echo "$p1_gpt_model" | sed 's|/|_|g; s|-|_|g')
    log_file="../experiment_outputs/job${JOB_ID}_exp${exp_count}_iter${iteration}_${recipe}_${safe_p0}_vs_${safe_p1}_${timestamp}.log"
    
    mkdir -p ../experiment_outputs
    
    # Write experiment metadata
    {
        echo "=== Job $JOB_ID - Experiment $exp_count ==="
        echo "Iteration: $iteration"
        echo "Recipe: $recipe"
        echo "P0 Model: $p0_model ($p0_gpt_model)"
        echo "P1 Model: $p1_model ($p1_gpt_model)"
        echo "P0 Server: $p0_server"
        echo "P1 Server: $p1_server"
        echo "Start Time: $(date)"
        echo "============================================"
    } > "$log_file"
    
    # Run experiment
    if timeout 3600 srun --nodes=1 --ntasks=1 python main.py \
            --order "${recipe}" \
            --temperature 0.7 \
            --p0_gpt_model "$p0_gpt_model" \
            --p1_gpt_model "$p1_gpt_model" \
            --p0_model_dirname "$p0_dir" \
            --p1_model_dirname "$p1_dir" \
            --p0_local_server_api "$p0_server" \
            --p1_local_server_api "$p1_server" >> "$log_file" 2>&1; then
        echo "✅ [$JOB_ID:$exp_count] SUCCESS: $p0_model vs $p1_model on $recipe"
    else
        exit_code=$?
        if [[ $exit_code == 124 ]]; then
            echo "⏰ [$JOB_ID:$exp_count] TIMEOUT: $p0_model vs $p1_model on $recipe"
        else
            echo "❌ [$JOB_ID:$exp_count] FAILED: $p0_model vs $p1_model on $recipe (exit code: $exit_code)"
        fi
    fi
    
    # Flush KV-cache
    flush_cache "${p0_server%/v1}"
    flush_cache "${p1_server%/v1}"
    
    # Update log file
    {
        echo "============================================"
        echo "End Time: $(date)"
        echo "Exit Code: ${exit_code:-0}"
    } >> "$log_file"
    
done < "$EXPERIMENT_FILE"

echo "🔍 DEBUG: Finished while loop, processed $exp_count experiments"
echo "🎯 Job $JOB_ID completed all $total_experiments experiments!"

# Cleanup - remove the experiment file
echo "🧹 Cleaning up experiment file: $EXPERIMENT_FILE"
rm -f "$EXPERIMENT_FILE"
