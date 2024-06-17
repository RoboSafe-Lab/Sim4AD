#!/bin/bash
#SBATCH --job-name=bc
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=50G
#SBATCH --nice
#SBATCH --gres=gpu:3

export PYTHONPATH=$HOME/Sim4AD/
source venv/bin/activate
python3 baselines/rl.py