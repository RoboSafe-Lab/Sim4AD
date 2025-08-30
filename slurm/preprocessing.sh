#!/bin/bash
#
#SBATCH --job-name=PREPROCESSING
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=144:00:00
#SBATCH --mem-per-cpu=30G
#
#
### You need to change the array indices depending on the scenario.
### For bendplatz and frankenburg use: --array=0-10
### For heckstrasse use: --array=0-2
### For round leave as it is.
#
###SBATCH --array=0-13%14
#
#SBATCH --nice

### Adapt the following lines to reflect the paths on your server.
### To create a new environment, first run: `conda create --name OGRIT python=3.8`
source ~/anaconda3/etc/profile.d/conda.sh
conda activate sim4ad

export PYTHONPATH=$HOME/Sim4AD/

python ~/Sim4AD/preprocessing.py
