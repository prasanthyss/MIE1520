#!/bin/bash
#SBATCH --job-name=anli
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH -w overture --gres=gpu:1

# task-1
python3 ./fine_tuning.py --model BERT-Base --lr 1e-6 --dataset ANLI --n_epochs 15
python3 ./make_plot.py --file '../logs/finetune_bert-base-uncased_anli.txt'
