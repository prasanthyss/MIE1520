#!/bin/bash
#SBATCH --job-name=blarge
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# task-1
python3 ./fine_tuning.py --model BERT-Large --dataset SST --n_epochs 2
python3 ./make_plot.py --file '../logs/finetune_bert-large-uncased_sst2.txt'