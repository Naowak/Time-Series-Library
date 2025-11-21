#!/bin/bash
#SBATCH --job-name=classification
#SBATCH --output=logs/classif_%N_%j.out
#SBATCH --error=logs/classif_%N_%j.err
#SBATCH -C a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread
#SBATCH --account=rvc@a100
#SBATCH --qos=qos_gpu_a100-t3

# Charger les modules nécessaires
module purge
module load arch/a100
module load miniforge/24.9.0
conda deactivate
# conda activate 
source tsl_venv/bin/activate
set -x

srun --exclusive --ntasks=1 bash scripts/classification/DT.sh

wait