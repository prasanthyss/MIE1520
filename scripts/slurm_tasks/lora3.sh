#!/bin/bash
#SBATCH --job-name=rbase_anli
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# task-1
python3 ./lora.py --model RoBERTa-Base --dataset ANLI --n_epochs 10