#!/bin/bash
#SBATCH --job-name=rlarge
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH -w overture --gres=gpu:1

# task-1
python3 ./fine_tuning.py --model RoBERTa-Large --lr 1e-6 --dataset SST --n_epochs 6
python3 ./make_plot.py --file '../logs/finetune_roberta-large_sst2.txt'
