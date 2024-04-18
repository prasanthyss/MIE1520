#!/bin/bash
#SBATCH --job-name=rl_anli
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# ANLI
python3 ./lora.py --model RoBERTa-Large --dataset ANLI --n_epochs 25

# SST
#python3 ./lora.py --model RoBERTa-Large --dataset SST --n_epochs 6
