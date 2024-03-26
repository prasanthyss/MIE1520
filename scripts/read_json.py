# %% [markdown]
# ## Convert .txt to .json

# %%
import os
import json

def replace_quotes(line):
    # Replace single quotes with double quotes
    return line.replace("'", '"')

def process_txt_files(folder_path):
    # Iterate through files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                # Read the second line from the file
                lines = file.readlines()
                if len(lines) >= 2:
                    second_line = lines[1].strip()
                    second_line = replace_quotes(second_line)
                    line = second_line

            json_file = os.path.splitext(file_path)[0] + '.json'
            with open(json_file, 'w') as file:
                # Read the second line from the file
                file.write(line)

folder_path = os.path.join(os.path.dirname(os.getcwd()), 'logs')
process_txt_files(folder_path)

# %% [markdown]
# ## Read Json Files

# %%

def collect_data(finetune_type):
    """
    go through all the 'task.json' files 
    and collect best train and test accuracies
    """

    def read_json_file(file_path):
        # Open the JSON file for reading
        with open(file_path, 'r') as file:
            # Load the JSON data
            data = json.load(file)
        return data

    def collect_accuracy(json_data):
        train_accuracies = []
        test_accuracies = []
        for data in json_data:
            if 'eval_train_accuracy' in data:
                train_accuracies.append((data['eval_train_accuracy'], data['eval_train_loss'], data['epoch']))
            if 'eval_test_accuracy' in data:
                test_accuracies.append((data['eval_test_accuracy'], data['eval_test_loss'], data['epoch']))
        
        train_accuracies.sort()
        test_accuracies.sort()
        return train_accuracies, test_accuracies
    
    
    folderpath = "../logs"
    result_dict = {}


    # tasks
    tasks = ['bert-base-uncased_anli', 'bert-large-uncased_anli', \
                'roberta-base_anli', 'roberta-large_anli', 'roberta-large_mrpc',\
                    'roberta-large_qqp', 'roberta-large_sst2']
    
    for task in tasks:
      file_path = os.path.join(folderpath, '_'.join([finetune_type, task+'.json']))
      if (os.path.exists(file_path)):
         json_data = read_json_file(file_path)
         train, test = collect_accuracy(json_data)
         best_train = train[0][0]
         best_test = test[0][0]
         result_dict[task] = {'train': best_train, 'test': best_test}

    return result_dict

# %%



