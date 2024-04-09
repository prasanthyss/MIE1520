"""
If you run the file, it will go through logs dir and 
collects the second line into a json file.
Some helper functions are also written to use in lora.py
"""

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
            process_text_file(file_path)

def process_text_file(file_path):
    filename = os.path.basename(file_path)
    if filename.endswith(".txt"):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            # Read the last line containing logs from the .txt file
            line_to_write = ""
            for line in lines:
                if(line[0] == '['):
                    line_to_write = line
            line_to_write = line_to_write.strip()
            line_to_write = replace_quotes(line_to_write)

        json_file = os.path.splitext(file_path)[0] + '.json'
        # write the line to a json file
        with open(json_file, 'w') as file:
            file.write(line_to_write)

        return json_file

# %% [markdown]
# ## Read Json Files

# %%

def collect_data(finetune_type, task):
    """
    go through all the 'task.json' files 
    and collect best train and test accuracies
    """

    def read_json_file(json_file):
        # Open the JSON file for reading
        with open(json_file, 'r') as file:
            # Load the JSON data
            data = json.load(file)
        return data

    def collect_accuracy(json_data):
        train_accuracies = []
        test_accuracies = []
        for data in json_data:
            if 'eval_train_accuracy' in data:
                train_accuracies.append(data['eval_train_accuracy'])
            if 'eval_test_accuracy' in data:
                test_accuracies.append(data['eval_test_accuracy'])
        
        return train_accuracies, test_accuracies    
    
    folderpath = "../logs"
    result_dict = {}
 
    file_path = os.path.join(folderpath, '_'.join([finetune_type, task+'.txt']))
    if (os.path.exists(file_path)):
        json_file = process_text_file(file_path)
        json_data = read_json_file(json_file)
        train_accuracies, test_accuracies = collect_accuracy(json_data)
        best_train = max(train_accuracies)
        best_test = max(test_accuracies)
        result_dict[task] = {'train': best_train, 'test': best_test}

    return result_dict

# %%



