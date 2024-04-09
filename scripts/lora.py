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
        self.reached_accuracy = False

        self.reached_90_train_acc = False
        self.reached_90_test_acc = False

        self.reached_95_train_acc = False
        self.reached_95_test_acc = False

        self.reached_99_train_acc = False
        self.reached_99_test_acc = False


    def on_evaluate(self, args, state, control, metrics, **kwargs):
        """ 
        call back called on evaluate.
        stop training when we reach 99% train and test accuracy or reach max_epochs.
        """

        if('eval_train_accuracy' in metrics):
            metric_key = 'eval_train_accuracy'
            self.reached_90_train_acc = (metrics[metric_key] >= 0.90*self.train_accuracy)
            self.reached_95_train_acc = (metrics[metric_key] >= 0.95*self.train_accuracy)
            self.reached_99_train_acc = (metrics[metric_key] >= 0.99*self.train_accuracy)
        elif('eval_test_accuracy' in metrics):
            metric_key = 'eval_test_accuracy'
            self.reached_90_test_acc = (metrics[metric_key] >= 0.90*self.test_accuracy)
            self.reached_95_test_acc = (metrics[metric_key] >= 0.95*self.test_accuracy)
            self.reached_99_test_acc = (metrics[metric_key] >= 0.99*self.test_accuracy)
        
        self.reached_accuracy = (self.reached_99_train_acc and self.reached_99_test_acc)
        control.should_training_stop = (self.reached_accuracy or metrics['epoch'] >= self.num_epochs)

class PEFT(FineTune):
    def __init__(self, model_path, dataset_dict, train_acc, test_acc):
        super().__init__(model_path, dataset_dict)

        # set the log file
        log_dir = os.path.join(os.path.dirname(os.getcwd()), 'logs')
        self.log_file = os.path.join(log_dir, '_'.join(['lora', os.path.basename(model_path), 
                                                       os.path.basename(dataset_dict['path']+'.txt')]))
        self.train_acc = train_acc
        self.test_acc = test_acc
    
        self.log_lines = []
        
    def train(self, num_epochs=6, lr=None, eval_steps=None):
        lora_ranks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        callback = AccuracyStoppingCallback(self.train_acc, self.test_acc, num_epochs)

        self.logged_90_train = False
        self.logged_90_test = False

        self.logged_95_train = False
        self.logged_95_test = False

        self.logged_99_train = False
        self.logged_99_test = False

        def append_to_logs():
            """
            Helper function to write output to logs.
            """
            
            self.log_lines.append(str(datetime.now()) + " lora_rank: " + str(lora_rank) + " num_params: " + 
                                    str(model.get_nb_trainable_parameters()))
            self.log_lines.append(str(self.trainer.state.log_history) + "\n\n")
    
            # write first ranks to reach given accuracies at the beginning of the log file.
            if (not self.logged_90_train and callback.reached_90_train_acc):
                self.log_lines = ["90_train: " + str(datetime.now()) + " lora_rank: " + str(lora_rank) +  " num_params: " + 
                                    str(model.get_nb_trainable_parameters())] + self.log_lines
                self.logged_90_train = True
        
            if (not self.logged_90_test and callback.reached_90_test_acc):
                self.log_lines = ["90_test: " + str(datetime.now()) + " lora_rank: " + str(lora_rank) +  " num_params: " + 
                                    str(model.get_nb_trainable_parameters())] + self.log_lines
                self.logged_90_test = True

            if (not self.logged_95_train and callback.reached_95_train_acc):
                self.log_lines = ["95_train: " + str(datetime.now()) + " lora_rank: " + str(lora_rank) +  " num_params: " + 
                                    str(model.get_nb_trainable_parameters())] + self.log_lines
                self.logged_95_train = True
        
            if (not self.logged_95_test and callback.reached_95_test_acc):
                self.log_lines = ["95_test: " + str(datetime.now()) + " lora_rank: " + str(lora_rank) +  " num_params: " + 
                                    str(model.get_nb_trainable_parameters())] + self.log_lines
                self.logged_95_test = True
                
            if (not self.logged_99_train and callback.reached_99_train_acc):
                self.log_lines = ["99_train: " + str(datetime.now()) + " lora_rank: " + str(lora_rank) +  " num_params: " + 
                                    str(model.get_nb_trainable_parameters())] + self.log_lines
                self.logged_99_train = True
                
            if (not self.logged_99_test and callback.reached_99_test_acc):
                self.log_lines = ["99_test: " + str(datetime.now()) + " lora_rank: " + str(lora_rank) +  " num_params: " + 
                                    str(model.get_nb_trainable_parameters())] + self.log_lines
                self.logged_99_test = True
            

        for lora_rank in lora_ranks:
            if (callback.reached_accuracy):
                break
            
            print(f"Training with lora_rank: {lora_rank}")
            config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=lora_rank, lora_alpha=32)
            model = get_peft_model(self.model, config)
            training_args = TrainingArguments(output_dir="../logs",
                                          num_train_epochs=num_epochs,
                                          evaluation_strategy="epoch",
                                          weight_decay=0.01,
                                          save_strategy="no")

            if eval_steps is not None:
                training_Args.evaluation_strategy="steps"
                training_Args.eval_steps=eval_steps

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
                compute_metrics=compute_metrics,
                callbacks=[callback])

            self.trainer.train()
            # append logs to log lines
            append_to_logs()

        def write_logs():
            with open(self.log_file, "a") as file:
                file.write(str(datetime.now())+"\n")
                file.write(f"training_accuracy: {self.train_acc}, test_accuracy: {self.test_acc}\n")
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
                    help="Steps to evaluate model", default=1000)

def main():
    args = parser.parse_args()

    model_path = args.model
    dataset_path = args.dataset
    num_epochs = args.n_epochs
    lr = args.lr
    eval_steps = args.eval_steps

    results_dict = collect_data("finetune")
    task = '_'.join([os.path.basename(models_dict[model_path]), os.path.basename(datasets_dict[dataset_path]['path'])])
    train_acc, test_acc = results_dict[task]['train'], results_dict[task]['test']

    model = PEFT(models_dict[model_path], datasets_dict[dataset_path], train_acc, test_acc)
    model.train(num_epochs=num_epochs, lr=lr, eval_steps=eval_steps)

if __name__ == "__main__":
    main()
