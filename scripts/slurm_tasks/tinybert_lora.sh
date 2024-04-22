#!/bin/bash
#SBATCH --job-name=tb_qqp
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# MRPC
#python3 ./lora.py --model Tiny-BERT --dataset MRPC --n_epochs 20

# QQP
python3 ./lora.py --model Tiny-BERT --dataset QQP --n_epochs 20

# ANLI
#python3 ./lora.py --model Tiny-BERT --dataset ANLI --n_epochs 25

# SST
#python3 ./lora.py --model Tiny-BERT --dataset SST --n_epochs 6
