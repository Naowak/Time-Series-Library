#!/bin/bash
#SBATCH --job-name=imputation
#SBATCH --output=logs/imputation_%N_%j.out
#SBATCH --error=logs/imputation_%N_%j.err
#SBATCH -C a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread
#SBATCH --account=rvc@a100
#SBATCH --qos=qos_gpu_a100-t3

# Charger les modules nécessaires
module purge
module load arch/a100
#module load miniforge/24.9.0
#conda deactivate
# conda activate 
module load python-3.11.5
source mvenv/bin/activate
set -x

# Define triton cache to avoid user disk limitation
export TRITON_CACHE_DIR=./triton_cache/
mkdir -p "$TRITON_CACHE_DIR"
echo "Triton cache directory: $TRITON_CACHE_DIR"

# Define an array of arguments for the different runs
ARGS=(
    "scripts/imputation/ECL_script/DT.sh"
    "scripts/imputation/ETT_script/DT.sh"
    "scripts/imputation/Weather_script/DT1.sh"
    "scripts/imputation/Weather_script/DT2.sh"
)

# Boucle pour lancer chaque instance du script avec les arguments correspondants
for ((i=0; i<4; i++)); do
    srun --exclusive --ntasks=1 --gpus=1 --export=ALL,CUDA_VISIBLE_DEVICES=$i bash ${ARGS[i]} &
done

wait
