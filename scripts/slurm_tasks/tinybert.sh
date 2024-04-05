#!/bin/bash
#SBATCH --job-name=tinybert
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# task-1
python3 ./fine_tuning.py --model Tiny-BERT --dataset SST --n_epochs 6
python3 ./make_plot.py --file '../logs/finetune_dynamic_tinybert_sst2.txt'
