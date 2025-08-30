#!/bin/bash
#SBATCH --job-name=SAC
#SBATCH --array=0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3-24:00:00
#SBATCH --mem-per-cpu=100G
#SBATCH --nice

source ~/anaconda3/etc/profile.d/conda.sh
conda activate sim4ad

export PYTHONPATH=$HOME/Sim4AD/

# Define parameter combinations
PARAMS=(
  "--cluster Aggressive"
)

# Run the appropriate command based on SLURM_ARRAY_TASK_ID
python ~/Sim4AD/baselines/sac/model.py # ${PARAMS[$SLURM_ARRAY_TASK_ID]}

