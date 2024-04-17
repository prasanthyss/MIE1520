#!/bin/bash
#SBATCH --job-name=tinybert
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# SST
# python3 ./fine_tuning.py --model Tiny-BERT --dataset SST --n_epochs 6
# python3 ./make_plot.py --file '../logs/finetune_bert-base-uncased_sst2.txt'

# MRPC
#python3 ./fine_tuning.py --model Tiny-BERT --dataset MRPC --n_epochs 6
#python3 ./make_plot.py --file '../logs/finetune_bert-base-uncased_mrpc.txt'

# ANLI
python3 ./fine_tuning.py --model Tiny-BERT --dataset ANLI --n_epochs 15
python3 ./make_plot.py --file '../logs/finetune_bert-base-uncased_anli.txt'

# QQP
#python3 ./fine_tuning.py --model Tiny-BERT --dataset QQP --n_epochs 6
#python3 ./make_plot.py --file '../logs/finetune_bert-base-uncased_qqp.txt'
