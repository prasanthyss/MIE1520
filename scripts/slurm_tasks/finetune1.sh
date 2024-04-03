#!/bin/bash
#SBATCH --job-name=bbase_anli
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# task-1
python3 ./fine_tuning.py --model BERT-Base --dataset SST2 --n_epochs 2
python3 ./make_plot.py --file '../logs/finetune_bert-base-uncased_sst2.txt'