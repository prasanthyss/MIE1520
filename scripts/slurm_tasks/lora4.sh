#!/bin/bash
#SBATCH --job-name=lr5_anli
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# MRPC
#python3 ./lora.py --model RoBERTa-Large --dataset MRPC --lr 1e-6 --n_epochs 20

# ANLI
python3 ./lora.py --model RoBERTa-Large --dataset ANLI --lr 1e-7 --n_epochs 25

# SST
#python3 ./lora.py --model RoBERTa-Large --dataset SST --n_epochs 6
