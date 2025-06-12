#!/bin/bash
#SBATCH --job-name=qwen7b_test
#SBATCH --output=slurm/%j.log
#SBATCH --cpus-per-task=1
#SBATCH --mem=1GB
#SBATCH --time=00:01:30
#SBATCH --nodelist=airl.ist.berkeley.edu

echo "Running on host: $(hostname)"
ss -tuln | grep 8000 || echo "Nothing on port 8000"

curl -X POST http://0.0.0.0:8000/v1/completions \
     -H "Content-Type: application/json" \
     -d '{
           "model": "Qwen/Qwen2.5-7B-Instruct",
           "prompt": "Once upon a time, in a land far away, ",
           "max_tokens": 64,
           "temperature": 0.7
         }'
