from fine_tuning import FineTune, models_dict, datasets_dict
from peft import LoraConfig, TaskType
from peft import get_peft_model
from read_json import collect_data
from transformers import TrainingArguments, Trainer
import evaluate
from datetime import datetime
import numpy as np

from transformers import TrainerCallback
import os

class AccuracyStoppingCallback(TrainerCallback):
    def __init__(self, train_accuracy, test_accuracy, num_epochs):
        self.train_accuracy = train_accuracy
        self.test_accuracy = test_accuracy
        self.num_epochs = num_epochs
        self.callback_called = False

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # stop if we reached the desired accuracy or max_epochs
        if (metrics['eval_train_accuracy'] >= 0.9*self.train_accuracy and \
           metrics['eval_test_accuracy'] >= 0.9*self.test_accuracy) or metrics['epoch'] >= self.num_epochs:
            control.should_training_stop = True
            self.callback_called = True

class PEFT(FineTune):
    def __init__(self, model_path, dataset_dict, train_acc, test_acc):
        super().__init__(model_path, dataset_dict)

        # set the log file
        log_dir = os.path.join(os.path.dirname(os.getcwd()), 'logs')
        self.log_file = os.path.join(log_dir, '_'.join(['lora', os.path.basename(model_path), 
                                                       os.path.basename(dataset_dict['path']+'.json')]))
        self.train_acc = train_acc
        self.test_acc = test_acc
        
        
    def train(self, num_epochs=6):
        lora_ranks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        callback = AccuracyStoppingCallback(self.train_acc, self.test_acc, num_epochs)
        for lora_rank in lora_ranks:
            if (callback.callback_called):
                break
            config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=lora_rank, lora_alpha=32)
            model = get_peft_model(self.model, config)
            training_args = TrainingArguments(output_dir="logs", 
                                            num_train_epochs=num_epochs, 
                                            evaluation_strategy="epoch",
                                            save_strategy="no")

            accuracy = evaluate.load("accuracy")
            def compute_metrics(eval_pred):
                predictions, labels = eval_pred
                predictions = np.argmax(predictions, axis=1)
                return accuracy.compute(predictions=predictions, references=labels)

            self.trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=self.tokenized_dataset['train'],
                eval_dataset=self.tokenized_dataset,
                compute_metrics=compute_metrics,
                callbacks=[callback]
            )

            self.trainer.train()

        def write_logs():
            with open(self.log_file, "a") as file:
                file.write(str(datetime.now())+ " lora_rank " + str(lora_rank) + "\n")
                file.write(str(self.trainer.state.log_history) + "\n\n")
            print(f"Results are appended to {self.log_file}")
    
        # save results at the end
        write_logs()


import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, choices=['BERT-Base', 'BERT-Large', 'RoBERTa-Base', 'RoBERTa-Large'], 
                    help="Please pass the model you want to train", required=True)
parser.add_argument('--dataset', type=str, choices=['MRPC', 'QQP', 'SST', 'ANLI'], 
                    help="Please specify the dataset to finetune", required=True)
parser.add_argument('--n_epochs', type=int, 
                    help="Default epochs is 6", default=6)

def main():
    args = parser.parse_args()

    model_path = args.model
    dataset_path = args.dataset
    num_epochs = args.n_epochs

    results_dict = collect_data("finetune")
    task = '_'.join([os.path.basename(models_dict[model_path]), os.path.basename(datasets_dict[dataset_path]['path'])])
    train_acc, test_acc = results_dict[task]['train'], results_dict[task]['test']

    model = PEFT(models_dict[model_path], datasets_dict[dataset_path], train_acc, test_acc)
    model.train(num_epochs=num_epochs)

if __name__ == "__main__":
    main()