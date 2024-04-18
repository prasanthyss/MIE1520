#!/bin/bash
#SBATCH --job-name=tb_anli
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# ANLI
python3 ./lora.py --model Tiny-BERT --dataset ANLI --n_epochs 25

# SST
#python3 ./lora.py --model Tiny-BERT --dataset SST --n_epochs 6
