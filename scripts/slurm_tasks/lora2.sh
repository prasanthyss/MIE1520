#!/bin/bash
#SBATCH --job-name=bl_anli
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH -w overture --gres=gpu:1

# ANLI
python3 ./lora.py --model BERT-Large --dataset ANLI --n_epochs 25

# SST
#python3 ./lora.py --model BERT-Large --dataset SST --n_epochs 6
