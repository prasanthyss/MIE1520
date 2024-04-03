#!/bin/bash
#SBATCH --job-name=rbase_anli
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# task-1
python3 ./fine_tuning.py --model RoBERTa-Base --lr 1e-2 --dataset ANLI --n_epochs 2
python3 ./make_plot.py --file '../logs/finetune_roberta-base_anli.txt'
