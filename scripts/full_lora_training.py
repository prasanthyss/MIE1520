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


class PEFT(FineTune):
    def __init__(self, model_path, dataset_dict):
        super().__init__(model_path, dataset_dict)

        # set the log file
        log_dir = os.path.join(os.path.dirname(os.getcwd()), 'logs')
        self.log_file = os.path.join(log_dir, '_'.join(['full_lora', os.path.basename(model_path), 
                                                       os.path.basename(dataset_dict['path']+'.txt')]))
        self.log_lines = []
        
    def train(self, num_epochs=6, lr=None, eval_steps=None):
        lora_ranks = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        lora_accuracies = []

        def append_to_log_lines():
            """
            Helper function to write output to logs.
            """
            
            self.log_lines.append(str(datetime.now()) + " lora_rank: " + str(lora_rank) + " num_params: " + 
                                    str(model.get_nb_trainable_parameters()))
            self.log_lines.append(str(self.trainer.state.log_history) + "\n\n")
            train_accuracies = [result_dict['eval_train_accuracy'] for result_dict in self.trainer.state.log_history \
                                if 'eval_train_accuracy' in result_dict]
            lora_accuracies.append(max(train_accuracies))
            

        for lora_rank in lora_ranks:
            
            print(f"Training with lora_rank: {lora_rank}")
            config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=lora_rank, lora_alpha=32)
            model = get_peft_model(self.model, config)
            training_args = TrainingArguments(output_dir="../logs",
                                          num_train_epochs=num_epochs,
                                          evaluation_strategy="epoch",
                                          #weight_decay=0.01,
                                          save_strategy="no")

            if eval_steps is not None:
                training_args.evaluation_strategy="steps"
                training_args.eval_steps=eval_steps

            if lr is not None:
                training_args.learning_rate=lr

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
                compute_metrics=compute_metrics)

            self.trainer.train()
            # append logs to log lines
            append_to_log_lines()
        
        # write lora accuracies to the log_lines
        self.log_lines = ['[' + ', '.join([str(acc) for acc in lora_accuracies]) + ']'] + self.log_lines

        def write_logs():
            with open(self.log_file, "a") as file:
                file.write(str(datetime.now())+"\n")
                lines = "\n".join(self.log_lines)
                file.write(lines)
            print(f"Results are appended to {self.log_file}")
    
        # save results at the end
        write_logs()


import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, choices=['Tiny-BERT', 'BERT-Base', 'BERT-Large', 'RoBERTa-Base', 'RoBERTa-Large'], 
                    help="Please pass the model you want to train", required=True)
parser.add_argument('--dataset', type=str, choices=['MRPC', 'QQP', 'SST', 'ANLI', 'YELP', 'MNLI'], 
                    help="Please specify the dataset to finetune", required=True)
parser.add_argument('--n_epochs', type=int, 
                    help="Default epochs is 6", default=6)
parser.add_argument('--lr', type=float, 
                    help="Learning rate to train the model.", default=5e-5)
parser.add_argument('--eval_steps', type=int, 
                    help="Steps to evaluate model", default=None)

def main():
    args = parser.parse_args()

    model_path = args.model
    dataset_path = args.dataset
    num_epochs = args.n_epochs
    lr = args.lr
    eval_steps = args.eval_steps

    task = '_'.join([os.path.basename(models_dict[model_path]), os.path.basename(datasets_dict[dataset_path]['path'])])
    results_dict = collect_data("finetune", task)

    model = PEFT(models_dict[model_path], datasets_dict[dataset_path])
    model.train(num_epochs=num_epochs, lr=lr, eval_steps=eval_steps)

if __name__ == "__main__":
    main()
