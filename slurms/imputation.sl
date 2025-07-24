#!/bin/bash
#SBATCH --job-name=imputation
#SBATCH --output=logs/imputation_%N_%j.out
#SBATCH --error=logs/imputation_%N_%j.err
#SBATCH -C a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --gres=gpu:3
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

# Define an array of arguments for the different runs
ARGS=(
    "scripts/imputation/ECL_script/EST.sh"
    "scripts/imputation/ETT_script/EST.sh"
    "scripts/imputation/Weather_script/EST.sh"
)

# Boucle pour lancer chaque instance du script avec les arguments correspondants
for ((i=0; i<3; i++)); do
    srun --exclusive --ntasks=1 bash ${ARGS[i]} &
done

wait