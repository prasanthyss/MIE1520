#!/bin/bash
#SBATCH --job-name=bl
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# MRPC
python3 ./full_lora_training.py --model BERT-Large --dataset MRPC --n_epochs 20
