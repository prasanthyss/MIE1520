#!/bin/bash
#SBATCH --job-name=rb_anli
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# ANLI
python3 ./lora.py --model RoBERTa-Base --dataset ANLI --lr 5e-6 --n_epochs 25

# SST
#python3 ./lora.py --model RoBERTa-Base --dataset SST --n_epochs 6
