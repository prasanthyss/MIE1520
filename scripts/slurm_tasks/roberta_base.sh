#!/bin/bash
#SBATCH --job-name=rbase
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# SST
# python3 ./fine_tuning.py --model RoBERTa-Base --lr 1e-7 --dataset SST --n_epochs 6

# MRPC
# python3 ./fine_tuning.py --model RoBERTa-Base --lr 1e-7 --dataset MRPC --n_epochs 6

# ANLI
python3 ./fine_tuning.py --model RoBERTa-Base --lr 1e-6 --dataset ANLI --n_epochs 30

# QQP
# python3 ./fine_tuning.py --model RoBERTa-Base --lr 1e-7 --dataset QQP --n_epochs 6
