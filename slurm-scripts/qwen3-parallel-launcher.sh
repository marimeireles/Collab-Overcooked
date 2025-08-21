#!/bin/bash
# Script to launch multiple parallel SLURM jobs for experiments

# Configuration
NUM_PARALLEL_JOBS=10
TOTAL_COMBINATIONS=33
TOTAL_RECIPES=5
ITERATIONS=2

# Calculate experiments per job
TOTAL_EXPERIMENTS=$((TOTAL_COMBINATIONS * TOTAL_RECIPES * ITERATIONS))
EXPERIMENTS_PER_JOB=$((TOTAL_EXPERIMENTS / NUM_PARALLEL_JOBS))
REMAINDER=$((TOTAL_EXPERIMENTS % NUM_PARALLEL_JOBS))

echo "🚀 Launching $NUM_PARALLEL_JOBS parallel SLURM jobs"
echo "📋 Total experiments: $TOTAL_EXPERIMENTS"
echo "📊 Experiments per job: $EXPERIMENTS_PER_JOB (with $REMAINDER remainder)"

# Create array of all experiment parameters
declare -a all_experiments=()

# Level 1 recipes
recipes=("baked_bell_pepper" "baked_sweet_potato" "boiled_egg" "boiled_mushroom" "boiled_sweet_potato")

# All model combinations (you'll need to match your randomized combinations)
combinations=(
    "QWEN3_32B QWEN3_14B"
    "QWEN3_8B QWEN3_32B"
    "QWEN3_0_6B QWEN3_0_6B"
    "QWEN3_8B QWEN3_8B"
    "QWEN3_1_7B QWEN3_4B"
    "QWEN3_32B QWEN3_8B"
    "QWEN3_14B QWEN3_1_7B"
    "QWEN3_4B QWEN3_14B"
    "QWEN3_1_7B QWEN3_8B"
    "QWEN3_14B QWEN3_4B"
    "QWEN3_14B QWEN3_14B"
    "QWEN3_4B QWEN3_4B"
    "QWEN3_32B QWEN3_1_7B"
    "QWEN3_14B QWEN3_8B"
    "QWEN3_32B QWEN3_4B"
    "QWEN3_1_7B QWEN3_1_7B"
    "QWEN3_4B QWEN3_1_7B"
    "QWEN3_8B QWEN3_1_7B"
    "QWEN3_32B QWEN3_32B"
    "QWEN3_0_6B QWEN3_8B"
    "QWEN3_0_6B QWEN3_14B"
    "QWEN3_0_6B QWEN3_32B"
    "QWEN3_0_6B QWEN3_1_7B"
    "QWEN3_0_6B QWEN3_4B"
    "QWEN3_1_7B QWEN3_32B"
    "QWEN3_1_7B QWEN3_14B"
    "QWEN3_4B QWEN3_32B"
    "QWEN3_4B QWEN3_8B"
    "QWEN3_8B QWEN3_14B"
    "QWEN3_8B QWEN3_4B"
    "QWEN3_14B QWEN3_32B"
    "QWEN3_14B QWEN3_0_6B"
    "QWEN3_32B QWEN3_0_6B"
)

# Build experiment list
exp_num=0
for iteration in {1..2}; do
    for recipe in "${recipes[@]}"; do
        for combo in "${combinations[@]}"; do
            read -r p0_model p1_model <<< "$combo"
            all_experiments+=("$iteration|$recipe|$p0_model|$p1_model")
            ((exp_num++))
        done
    done
done

echo "📝 Generated ${#all_experiments[@]} total experiments"

# Launch parallel jobs
for job_id in $(seq 1 $NUM_PARALLEL_JOBS); do
    start_idx=$(( (job_id - 1) * EXPERIMENTS_PER_JOB ))
    end_idx=$(( start_idx + EXPERIMENTS_PER_JOB - 1 ))
    
    # Handle remainder for last job
    if [[ $job_id -eq $NUM_PARALLEL_JOBS ]]; then
        end_idx=$((${#all_experiments[@]} - 1))
    fi
    
    echo "🎯 Launching job $job_id: experiments $start_idx to $end_idx"
    
    # Create job-specific experiment list
    job_experiments=("${all_experiments[@]:$start_idx:$((end_idx - start_idx + 1))}")
    
    # Write experiments to shared file for this job (use shared NAS storage)
    shared_dir="/nas/ucb/$USER/dev/Collab-Overcooked/slurm-scripts/job_data"
    mkdir -p "$shared_dir"
    temp_file="${shared_dir}/experiments_job_${job_id}.txt"
    printf '%s\n' "${job_experiments[@]}" > "$temp_file"
    
    # Submit SLURM job
    sbatch --job-name="qwen3-job-${job_id}" \
           --output="slurm/qwen3-job-${job_id}-%j.log" \
           --export="EXPERIMENT_FILE=${temp_file},JOB_ID=${job_id}" \
           qwen3-single-job.sh
done

echo "✅ Launched $NUM_PARALLEL_JOBS parallel jobs!"
echo "📊 Monitor with: squeue -u $USER"
echo "📄 Check logs in: slurm/qwen3-job-*"
