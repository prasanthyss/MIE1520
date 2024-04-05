#!/bin/bash
#SBATCH --job-name=rlarge
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH -w overture --gres=gpu:1

# task-1
python3 ./lora.py --model RoBERTa-Large --dataset SST --n_epochs 6
