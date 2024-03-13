#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import os
from datetime import datetime
import warnings


# In[5]:


datasets_dict = {'MRPC': {'path': "SetFit/mrpc", 'data': ['text1', 'text2'], 'split': 'train'},
                 'QQP': {'path': "SetFit/qqp", 'data': ['text1', 'text2'], 'split': 'train'},
                 'SST': {'path': "sst2", 'data': ['sentence'], 'split': 'train'},
                 'ANLI': {'path': "facebook/anli", 'data': ['premise', 'hypothesis'], 'split': 'train_r1'}}

models_dict = {'BERT-Base': "google-bert/bert-base-uncased",
               'BERT-Large': "google-bert/bert-large-uncased",
               'RoBERTa-Base': "FacebookAI/roberta-base",
               'RoBERTa-Large': "FacebookAI/roberta-large"}


# In[7]:


class FineTune():
    def __init__(self, model_path, dataset_dict):
        # Load the dataset
        print(f"Loading the dataset from {dataset_dict['path']}")
        self.dataset = load_dataset(dataset_dict['path'], split = dataset_dict['split'])
        self.data = dataset_dict['data']
        # Task is paraphrasing if we have more than one data column
        self.paraphrase = (len(self.data) != 1)

        print('Generating tokens for the dataset')
        self.tokenizer =  AutoTokenizer.from_pretrained(model_path)
        self.tokenized_dataset = self.tokenize()
        self.num_labels = len(set(self.dataset['label']))

        print(f'Loading the model from {model_path}')
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=self.num_labels)

        self.log_file = os.path.join(os.getcwd(), 'logs', os.path.basename(model_path)+'_'+os.path.basename(dataset_dict['path']+'.txt'))
        print(f'Metrics are written to {self.log_file}')

    def tokenize(self):
        if(self.paraphrase):
            prompt = "{text1}\nPARAPHRASE:\n{text2}"
            col1, col2 = self.data
            text_data = [prompt.format(text1=sentence1, text2=sentence2) for sentence1, sentence2 in zip(self.dataset[col1], self.dataset[col2])]
            self.dataset = self.dataset.add_column('text', text_data)
            self.data = 'text'
        else:
            self.data = self.data[0]

        def tokenize_function(examples):
            return self.tokenizer(examples[self.data], padding="max_length", truncation=True)

        tokenized_dataset = self.dataset.map(tokenize_function, batched=True)

        return tokenized_dataset

    def train(self, num_epochs=6):
        training_args = TrainingArguments(output_dir="logs", num_train_epochs=num_epochs)

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset
        )

        self.trainer.train()

        def write_logs():
          with open(self.log_file, "w") as file:
              file.write(str(datetime.now())+"\n")
              file.write(str(self.trainer.state.log_history) + "\n\n")

        write_logs()


# In[12]:


from peft import LoraConfig, TaskType
from peft import get_peft_model

class PEFT(FineTune):
    def __init__(self, model_path, dataset_dict, lora_rank=4):
        super().__init__(model_path, dataset_dict)
        self.rank = lora_rank
        self.config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=self.rank, lora_alpha=32)
        self.model = get_peft_model(self.model, self.config)


# In[13]:

model = FineTune(models_dict['BERT-Base'], datasets_dict['MRPC'])
model.train()




