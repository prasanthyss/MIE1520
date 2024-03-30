#!/bin/bash
#SBATCH --job-name=lora_95
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
# task-1
python3 ./lora.py --model BERT-Base --dataset ANLI --n_epochs 10
python3 ./lora.py --model BERT-Large --dataset ANLI --n_epochs 10
python3 ./lora.py --model RoBERTa-Base --dataset ANLI --n_epochs 10
python3 ./lora.py --model RoBERTa-Large --dataset ANLI --n_epochs 10

# task-2
python3 ./lora.py --model RoBERTa-Large --dataset MRPC --n_epochs 10
python3 ./lora.py --model RoBERTa-Large --dataset QQP --n_epochs 10
python3 ./lora.py --model RoBERTa-Large --dataset SST --n_epochs 10
