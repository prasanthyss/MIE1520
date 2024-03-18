# task-1
python3 ./fine_tuning.py --model BERT-Base --dataset ANLI --n_epochs 20
python3 ./fine_tuning.py --model BERT-Large --dataset ANLI --n_epochs 20
python3 ./fine_tuning.py --model RoBERTa-Base --dataset ANLI --n_epochs 20
python3 ./fine_tuning.py --model RoBERTa-Large --dataset ANLI --n_epochs 20

# task-2
python3 ./fine_tuning.py --model RoBERTa-Large --dataset MRPC --n_epochs 20
python3 ./fine_tuning.py --model RoBERTa-Large --dataset QQP --n_epochs 20
python3 ./fine_tuning.py --model RoBERTa-Large --dataset SST --n_epochs 20