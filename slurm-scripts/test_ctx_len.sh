#!/bin/bash
#SBATCH --job-name=qwen7b_test
#SBATCH --output=slurm/%j.log
#SBATCH --cpus-per-task=1
#SBATCH --mem=1GB
#SBATCH --time=00:01:30
#SBATCH --nodelist=gail.ist.berkeley.edu

# ======= Paths and environment =======
export MAMBA_ROOT_PREFIX=/nas/ucb/$USER/micromamba
export PATH=$MAMBA_ROOT_PREFIX/bin:$PATH
eval "$($MAMBA_ROOT_PREFIX/micromamba shell hook --shell bash)"
micromamba activate overcooked

pytest /nas/ucb/marimeireles/dev/Collab-Overcooked/slurm-scripts/test_context_window.py