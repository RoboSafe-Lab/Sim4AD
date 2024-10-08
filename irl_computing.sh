#!/bin/bash
#
#SBATCH --job-name=IRL_COMPUTATION
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=144:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH -p nodes
#
#
### You need to change the array indices depending on the scenario.
### For bendplatz and frankenburg use: --array=0-10
### For heckstrasse use: --array=0-2
### For round leave as it is.
#
#SBATCH --array=0-13%14
#
#SBATCH --nice

### Adapt the following lines to reflect the paths on your server.
### To create a new environment, first run: `conda create --name OGRIT python=3.8`
source ~/anaconda3/etc/profile.d/conda.sh
conda activate sim4ad

export PYTHONPATH=$HOME/Sim4AD/

### Change "frankenburg" to be the scenario you want the features for. Choose one of: bendplatz, frankenburg, heckstrasse, round
python ~/Sim4AD/feature_extraction_irl.py --map appershofen --clustering True --episode_idx $SLURM_ARRAY_TASK_ID 
