#!/bin/bash
#SBATCH --job-name=rlarge_sst
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# task-1
python3 ./fine_tuning.py --model RoBERTa-Large --dataset SST --n_epochs 10