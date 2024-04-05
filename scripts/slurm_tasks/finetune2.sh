#!/bin/bash
#SBATCH --job-name=blarge
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH -w overture --gres=gpu:1

# task-1
python3 ./fine_tuning.py --model BERT-Large --lr 5e-7 --dataset SST --n_epochs 6
python3 ./make_plot.py --file '../logs/finetune_bert-large-uncased_sst2.txt'
