#!/bin/bash
#SBATCH --job-name=tb_mrpc
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# SST
# python3 ./fine_tuning.py --model Tiny-BERT --dataset SST --n_epochs 6

# MRPC
python3 ./fine_tuning.py --model Tiny-BERT --dataset MRPC --n_epochs 6

# ANLI
#python3 ./fine_tuning.py --model Tiny-BERT --dataset ANLI --n_epochs 30

# QQP
#python3 ./fine_tuning.py --model Tiny-BERT --dataset QQP --n_epochs 6
#python3 ./make_plot.py --file '../logs/finetune_bert-base-uncased_qqp.txt'
