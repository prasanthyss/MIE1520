#!/bin/bash
#SBATCH --job-name=bl_mrpc
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# ANLI
python3 ./lora.py --model BERT-Large --dataset MRPC --n_epochs 20

# ANLI
#python3 ./lora.py --model BERT-Large --dataset ANLI --n_epochs 25

# SST
#python3 ./lora.py --model BERT-Large --dataset SST --n_epochs 6
