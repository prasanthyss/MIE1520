#!/bin/bash
#SBATCH --job-name=qqp
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH -w overture --gres=gpu:1

# task-1
python3 ./lora.py --model BERT-Base --eval_steps 100000 --dataset QQP --n_epochs 6
