#!/bin/bash
#SBATCH --job-name=anli
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH -w overture --gres=gpu:1

# task-1
python3 ./fine_tuning.py --model BERT-Base --dataset ANLI --n_epochs 6
python3 ./make_plot.py --file '../logs/finetune_bert-base-uncased_anli.txt'
