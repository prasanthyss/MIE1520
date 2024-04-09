#!/bin/bash

#SBATCH --job-name=yelp
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

python3 ./fine_tuning.py --model BERT-Base --dataset YELP --n_epochs 6
python3 ./make_plot.py --file '../logs/finetune_bert-base-uncased_yelp_review_full.txt'
