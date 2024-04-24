#!/bin/bash
#SBATCH --job-name=blarge_mrpc
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH  --gres=gpu:1

# SST
# python3 ./fine_tuning.py --model BERT-Large --lr 5e-7 --dataset SST --n_epochs 6
# python3 ./make_plot.py --file '../logs/finetune_bert-base-uncased_sst2.txt'

# MRPC
python3 ./fine_tuning.py --model BERT-Large --lr 1e-7 --dataset MRPC --n_epochs 30
# python3 ./make_plot.py --file '../logs/finetune_bert-base-uncased_mrpc.txt'

# ANLI
#python3 ./fine_tuning.py --model BERT-Large --lr 5e-7 --dataset ANLI --eval_steps 500 --n_epochs 30
#python3 ./make_plot.py --file '../logs/finetune_bert-base-uncased_anli.txt'

# QQP
# python3 ./fine_tuning.py --model BERT-Large --lr 5e-7 --dataset QQP --n_epochs 6
# python3 ./make_plot.py --file '../logs/finetune_bert-base-uncased_qqp.txt'
