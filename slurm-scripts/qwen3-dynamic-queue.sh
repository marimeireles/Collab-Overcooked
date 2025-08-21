#!/bin/bash
# Dynamic queue system - launches 10 jobs at a time, replacing finished ones

# Configuration
MAX_CONCURRENT_JOBS=10
EXPERIMENTS_PER_JOB=10  # Smaller batches for more dynamic scheduling

# Create experiment queue file
QUEUE_FILE="/tmp/experiment_queue_$$"
ACTIVE_JOBS_FILE="/tmp/active_jobs_$$"
COMPLETED_COUNT_FILE="/tmp/completed_count_$$"

# Initialize counters
echo "0" > "$COMPLETED_COUNT_FILE"
touch "$ACTIVE_JOBS_FILE"

# Level 1 recipes
recipes=("baked_bell_pepper" "baked_sweet_potato" "boiled_egg" "boiled_mushroom" "boiled_sweet_potato")

# All model combinations
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

# Build experiment queue
echo "🔨 Building experiment queue..."
for iteration in {1..2}; do
    for recipe in "${recipes[@]}"; do
        for combo in "${combinations[@]}"; do
            read -r p0_model p1_model <<< "$combo"
            echo "$iteration|$recipe|$p0_model|$p1_model" >> "$QUEUE_FILE"
        done
    done
done

total_experiments=$(wc -l < "$QUEUE_FILE")
echo "📋 Created queue with $total_experiments total experiments"
echo "🎯 Using batches of $EXPERIMENTS_PER_JOB experiments per job"
echo "🔄 Maintaining $MAX_CONCURRENT_JOBS concurrent jobs"

# Function to launch a job batch
launch_job_batch() {
    local batch_id=$1
    local temp_file="/tmp/batch_${batch_id}_$$"
    
    # Get next batch of experiments
    if ! head -n "$EXPERIMENTS_PER_JOB" "$QUEUE_FILE" > "$temp_file" 2>/dev/null; then
        return 1  # No more experiments
    fi
    
    # Remove these experiments from queue
    tail -n +$((EXPERIMENTS_PER_JOB + 1)) "$QUEUE_FILE" > "${QUEUE_FILE}.tmp" || touch "${QUEUE_FILE}.tmp"
    mv "${QUEUE_FILE}.tmp" "$QUEUE_FILE"
    
    # Check if batch has any experiments
    if [[ ! -s "$temp_file" ]]; then
        rm -f "$temp_file"
        return 1
    fi
    
    local batch_size=$(wc -l < "$temp_file")
    echo "🚀 Launching batch $batch_id with $batch_size experiments"
    
    # Submit job
    job_id=$(sbatch --job-name="qwen3-batch-${batch_id}" \
                   --output="slurm/qwen3-batch-${batch_id}-%j.log" \
                   --export="EXPERIMENT_FILE=${temp_file},BATCH_ID=${batch_id}" \
                   --parsable \
                   qwen3-single-job.sh)
    
    echo "$job_id|$batch_id|$batch_size" >> "$ACTIVE_JOBS_FILE"
    echo "✅ Launched batch $batch_id as SLURM job $job_id"
    
    return 0
}

# Function to check completed jobs and update counters
check_completed_jobs() {
    local new_active_jobs="/tmp/new_active_jobs_$$"
    touch "$new_active_jobs"
    
    while IFS='|' read -r job_id batch_id batch_size; do
        if [[ -z "$job_id" ]]; then continue; fi
        
        # Check if job is still running
        if squeue -j "$job_id" &>/dev/null; then
            # Job still running
            echo "$job_id|$batch_id|$batch_size" >> "$new_active_jobs"
        else
            # Job completed
            echo "✅ Batch $batch_id (job $job_id) completed"
            
            # Update completed count
            local completed=$(cat "$COMPLETED_COUNT_FILE")
            echo $((completed + batch_size)) > "$COMPLETED_COUNT_FILE"
            
            # Clean up temp file
            rm -f "/tmp/batch_${batch_id}_$$"
        fi
    done < "$ACTIVE_JOBS_FILE"
    
    mv "$new_active_jobs" "$ACTIVE_JOBS_FILE"
}

# Function to get current status
print_status() {
    local active_count=$(wc -l < "$ACTIVE_JOBS_FILE")
    local completed_count=$(cat "$COMPLETED_COUNT_FILE")
    local remaining_count=$(wc -l < "$QUEUE_FILE")
    local total_processed=$((completed_count + remaining_count))
    
    echo "📊 Status: $completed_count completed, $active_count active, $remaining_count queued (total: $total_processed)"
}

# Main execution loop
echo "🚀 Starting dynamic queue execution..."

batch_counter=1

# Launch initial batch of jobs
while [[ $(wc -l < "$ACTIVE_JOBS_FILE") -lt $MAX_CONCURRENT_JOBS ]] && [[ -s "$QUEUE_FILE" ]]; do
    if ! launch_job_batch "$batch_counter"; then
        break
    fi
    ((batch_counter++))
    sleep 2  # Brief pause between launches
done

print_status

# Main monitoring loop
while [[ -s "$QUEUE_FILE" ]] || [[ $(wc -l < "$ACTIVE_JOBS_FILE") -gt 0 ]]; do
    echo "🔄 Checking job status..."
    check_completed_jobs
    
    # Launch new jobs to fill slots
    while [[ $(wc -l < "$ACTIVE_JOBS_FILE") -lt $MAX_CONCURRENT_JOBS ]] && [[ -s "$QUEUE_FILE" ]]; do
        if ! launch_job_batch "$batch_counter"; then
            break
        fi
        ((batch_counter++))
        sleep 2
    done
    
    print_status
    
    # Wait before next check
    sleep 30
done

# Final status
check_completed_jobs
final_completed=$(cat "$COMPLETED_COUNT_FILE")

echo ""
echo "🎯 ============================================"
echo "🎯 ALL EXPERIMENTS COMPLETED!"
echo "🎯 ============================================"
echo "📋 Total experiments completed: $final_completed"
echo "📁 Logs saved in: slurm/qwen3-batch-*"
echo "📁 Results in: experiment_outputs/"

# Cleanup
rm -f "$QUEUE_FILE" "$ACTIVE_JOBS_FILE" "$COMPLETED_COUNT_FILE"
