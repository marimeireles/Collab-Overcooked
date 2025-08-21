#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=slurm/%j.log
#SBATCH --cpus-per-task=2
#SBATCH --mem=2GB
#SBATCH --time=00:05:00
#SBATCH --nodelist=airl.ist.berkeley.edu

curl -X POST localhost:10000/v1/completions \
     -H "Content-Type: application/json" \
     -d '{
           "model": "/nas/ucb/marimeireles/models/qwen2.5-32b",
           "prompt": "Once upon a time, in a land far away, ",
           "max_tokens": 64,
           "temperature": 0.7
         }'