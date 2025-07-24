#!/bin/bash
#SBATCH --job-name=long_term_forecast
#SBATCH --output=logs/long_term_forecast_%N_%j.out
#SBATCH --error=logs/long_term_forecast_%N_%j.err
#SBATCH -C a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --gres=gpu:5
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
    "scripts/long_term_forecast/ECL_script/EST.sh"
    "scripts/long_term_forecast/Exchange_script/EST.sh"
    "scripts/long_term_forecast/ILI_script/EST.sh"
    "scripts/long_term_forecast/Traffic_script/EST.sh"
    "scripts/long_term_forecast/Weather_script/EST.sh"
)

# Boucle pour lancer chaque instance du script avec les arguments correspondants
for ((i=0; i<5; i++)); do
    CUDA_VISIBLE_DEVICE=$i srun --exclusive --ntasks=1 --gpus=1 bash ${ARGS[i]} &
done

wait