#!/bin/bash
#SBATCH --job-name=bb_mrpc
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# task-1
python3 ./lora.py --model BERT-Base --dataset MRPC --n_epochs 6 --eval_steps 100
