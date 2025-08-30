#!/bin/bash
#
#SBATCH --job-name=IRL_COMPUTATION
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=144:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH -p nodes
#
#
#SBATCH --array=7-13%14
#
#SBATCH --nice

### Adapt the following lines to reflect the paths on your server.
### To create a new environment, first run: `conda create --name OGRIT python=3.8`
source ~/anaconda3/etc/profile.d/conda.sh
conda activate sim4ad

export PYTHONPATH=$HOME/Sim4AD/

python ~/Sim4AD/feature_extraction_irl.py --map appershofen --clustering True --episode_idx $SLURM_ARRAY_TASK_ID 
