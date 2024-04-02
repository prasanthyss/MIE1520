#!/bin/bash
#SBATCH --job-name=rlarge_mrpc
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# task-1
python3 ./fine_tuning.py --model RoBERTa-Large --dataset MRPC --n_epochs 10