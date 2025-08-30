#!/bin/bash
#SBATCH --job-name=bc
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=7G

#SBATCH --array=1-3%3
#SBATCH --nice

source ~/anaconda3/etc/profile.d/conda.sh
conda activate sim4ad

export PYTHONPATH=$HOME/Sim4AD/


if [ "$SLURM_ARRAY_TASK_ID" -ge 3 ]; then
    cluster="Aggressive"
elif [ "$SLURM_ARRAY_TASK_ID" -ge 2 ]; then
    cluster="Normal"
elif [ "$SLURM_ARRAY_TASK_ID" -ge 1 ]; then
    cluster="Cautious"
else
    cluster="All"
fi


python ~/Sim4AD/baselines/bc_baseline.py --cluster $cluster
