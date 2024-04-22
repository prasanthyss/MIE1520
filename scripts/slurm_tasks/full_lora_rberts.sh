#!/bin/bash
#SBATCH --job-name=rb
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# MRPC
python3 ./full_lora_training --model RoBERTa-Base --dataset MRPC --n_epochs 20