#!/bin/bash
#SBATCH --job-name=mie_project_lora
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
# task-1
python3 ./lora.py --model BERT-Base --dataset ANLI --n_epochs 20
python3 ./lora.py --model BERT-Large --dataset ANLI --n_epochs 20
python3 ./lora.py --model RoBERTa-Base --dataset ANLI --n_epochs 20
python3 ./lora.py --model RoBERTa-Large --dataset ANLI --n_epochs 20

# task-2
python3 ./lora.py --model RoBERTa-Large --dataset MRPC --n_epochs 20
python3 ./lora.py --model RoBERTa-Large --dataset QQP --n_epochs 20
python3 ./lora.py --model RoBERTa-Large --dataset SST --n_epochs 20
