#!/bin/bash
#SBATCH --job-name=ETT_forecast
#SBATCH --output=logs/ETT_forecast_%N_%j.out
#SBATCH --error=logs/ETT_forecast_%N_%j.err
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
module load miniforge/24.9.0
conda deactivate
# conda activate 
source tsl_venv/bin/activate
set -x

# Define an array of arguments for the different runs
ARGS=(
    "scripts/long_term_forecast/ETT_script/ESTh1.sh"
    "scripts/long_term_forecast/ETT_script/ESTh2.sh"
    "scripts/long_term_forecast/ETT_script/ESTm1.sh"
    "scripts/long_term_forecast/ETT_script/ESTm2.sh"
)

# Boucle pour lancer chaque instance du script avec les arguments correspondants
for ((i=0; i<4; i++)); do
    srun --exclusive --ntasks=1 --gpus=1 --export=ALL,CUDA_VISIBLE_DEVICES=$i bash ${ARGS[i]} &
done

wait
