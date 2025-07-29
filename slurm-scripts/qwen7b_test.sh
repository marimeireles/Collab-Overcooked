#!/bin/bash
#SBATCH --job-name=qwen7b_test
#SBATCH --output=slurm/%j.log
#SBATCH --cpus-per-task=1
#SBATCH --mem=1GB
#SBATCH --time=00:01:30
#SBATCH --nodelist=gail.ist.berkeley.edu

echo "Running on host: $(hostname)"
ss -tuln | grep 8070 || echo "Nothing on port 8070"

curl -X POST http://0.0.0.0:8070/v1/completions \
     -H "Content-Type: application/json" \
     -d '{
           "model": "Qwen/Qwen2.5-7B-Instruct",
           "prompt": "whats my name?",
           "max_tokens": 64,
           "temperature": 0.7
         }'
