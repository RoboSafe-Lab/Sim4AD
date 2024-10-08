#!/bin/bash
#
#SBATCH --job-name=IRL_TRAINING
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=144:00:00
#SBATCH --mem-per-cpu=5G
#
#
### You need to change the array indices depending on the scenario.
### For bendplatz and frankenburg use: --array=0-10
### For heckstrasse use: --array=0-2
### For round leave as it is.
#
#SBATCH --array=0-3
DRIVING_STYLES=("Aggressive" "Normal" "Cautious" "All")
CURRENT_DRIVING_STYLE=${DRIVING_STYLES[$SLURM_ARRAY_TASK_ID]}
echo "Running job with driving style: $CURRENT_DRIVING_STYLE"


#
#SBATCH --nice

### Adapt the following lines to reflect the paths on your server.
### To create a new environment, first run: `conda create --name OGRIT python=3.8`
source ~/anaconda3/etc/profile.d/conda.sh
conda activate sim4ad

export PYTHONPATH=$HOME/Sim4AD/

python ~/Sim4AD/training_irl.py --map appershofen --driving_style "$CURRENT_DRIVING_STYLE" 
