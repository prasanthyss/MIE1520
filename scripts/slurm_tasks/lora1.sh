#!/bin/bash
#SBATCH --job-name=bb_anli
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# MRPC
python3 ./lora.py --model BERT-Base --dataset MRPC --n_epochs 6

# ANLI
#python3 ./lora.py --model BERT-Base --dataset ANLI --n_epochs 25

# SST
#python3 ./lora.py --model BERT-Base --dataset SST --n_epochs 6

