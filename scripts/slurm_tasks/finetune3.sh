#!/bin/bash
#SBATCH --job-name=rbase
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# task-1
python3 ./fine_tuning.py --model RoBERTa-Base --dataset SST --n_epochs 2
python3 ./make_plot.py --file '../logs/finetune_roberta-base_sst2.txt'
