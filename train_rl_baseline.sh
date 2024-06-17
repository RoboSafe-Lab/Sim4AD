#!/bin/bash
#SBATCH --job-name=bc
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH --nice

export PYTHONPATH=$HOME/Sim4AD/

python ~/Sim4AD/baselines/rl.py