#!/bin/bash
#SBATCH --job-name=bbase
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# SST
python3 ./fine_tuning.py --model BERT-Base --dataset SST --n_epochs 6
python3 ./make_plot.py --file '../logs/finetune_bert-base-uncased_sst2.txt'

# MRPC
python3 ./fine_tuning.py --model BERT-Base --dataset MRPC --eval_steps 100 --dataset MRPC --n_epochs 6
python3 ./make_plot.py --file '../logs/finetune_bert-base-uncased_mrpc.txt'

# ANLI
python3 ./fine_tuning.py --model BERT-Base --dataset ANLI --lr 1e-6 --n_epochs 15
python3 ./make_plot.py --file '../logs/finetune_bert-base-uncased_anli.txt'

# QQP
python3 ./fine_tuning.py --model BERT-Base --dataset QQP --eval_steps 10000 --n_epochs 6
python3 ./make_plot.py --file '../logs/finetune_bert-base-uncased_qqp.txt'
