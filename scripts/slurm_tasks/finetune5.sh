#!/bin/bash
#SBATCH --job-name=mrpc
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# task-1
python3 ./fine_tuning.py --model BERT-Base --dataset MRPC --eval_steps 100 --n_epochs 6
python3 ./make_plot.py --file '../logs/finetune_bert-base-uncased_mrpc.txt'
