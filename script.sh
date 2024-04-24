#!/bin/bash
#SBATCH --job-name=bc
#
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=5
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:3
#
#SBATCH --array=0-3
#SBATCH --nice

source ~/anaconda3/etc/profile.d/conda.sh
conda activate sim4ad_310

export PYTHONPATH=$HOME/Sim4AD/


if [ "$SLURM_ARRAY_TASK_ID" -ge 3 ]; then
    cluster="Aggressive"
elif [ "$SLURM_ARRAY_TASK_ID" -ge 2 ]; then
    cluster="Normal"
elif [ "$SLURM_ARRAY_TASK_ID" -ge 1 ]; then
    cluster="Cautious"
else
    cluster="all"
fi


python ~/OGRIT/baselines/lstm/get_results.py --cluster $cluster