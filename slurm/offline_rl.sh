#!/bin/bash
#
#SBATCH --job-name=OFFLINE_RL
#
#SBATCH --ntasks=1
###SBATCH --cpus-per-task=1
#SBATCH --time=144:00:00
#SBATCH --mem-per-cpu=30G
#
#SBATCH --gres gpu:1
#SBATCH -p gpu
#
### You need to change the array indices depending on the scenario.
### For bendplatz and frankenburg use: --array=0-10
### For heckstrasse use: --array=0-2
### For round leave as it is.
#SBATCH --array=0
DRIVING_STYLES=("Aggressive")
CURRENT_DRIVING_STYLE=${DRIVING_STYLES[$SLURM_ARRAY_TASK_ID]}
### Print the selected driving style for debugging purposes
echo "Running job with driving style: $CURRENT_DRIVING_STYLE"
#
#SBATCH --nice

### Adapt the following lines to reflect the paths on your server.
### To create a new environment, first run: `conda create --name OGRIT python=3.8`
source ~/anaconda3/etc/profile.d/conda.sh
conda activate sim4ad

export PYTHONPATH=$HOME/Sim4AD/

### Change "frankenburg" to be the scenario you want the features for. Choose one of: bendplatz, frankenburg, heckstrasse, round
python ~/Sim4AD/sim4ad/offlinerlenv/td3bc_automatum.py --driving_style "$CURRENT_DRIVING_STYLE"
 
