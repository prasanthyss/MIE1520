#!/bin/bash
#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --mem=32G


python3 ./fine_tuning.py --model BERT-Base --dataset MRPC --n_epochs 20

