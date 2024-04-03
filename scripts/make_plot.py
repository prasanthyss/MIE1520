import os
import json
import matplotlib.pyplot as plt

def replace_quotes(line):
    # Replace single quotes with double quotes
    return line.replace("'", '"')

def process_txt_file(file_path):
    # Iterate through files in the folder
    folder_path, filename = os.path.dirname(file_path), os.path.basename(file_path)
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

def collect_data(file_path):
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
        train_loss = []
        train_epochs = []
        test_loss = []
        test_epochs = []
        for data in json_data:
            if 'eval_train_accuracy' in data:
                train_loss.append(data['eval_train_loss'])
                train_epochs.append(data['epoch'])
            if 'eval_test_accuracy' in data:
                test_loss.append(data['eval_test_loss'])
                test_epochs.append(data['epoch'])
        
        return train_loss, train_epochs, test_loss, test_epochs
    
    json_data = read_json_file(file_path)
    train_acc, train_epoch, test_acc, test_epoch = collect_accuracy(json_data)

    return train_acc, train_epoch, test_acc, test_epoch

def make_plot(train_loss, train_epochs, test_loss, test_epochs, plot_path):
    # Plot train vs test accuracies
    plt.plot(train_epochs, train_loss, marker='o', label='training loss')
    plt.plot(test_epochs, test_loss, marker='o', label='test loss')

    # Labeling axes
    plt.xlabel('epochs')
    plt.ylabel('loss')
    
    plt.legend()

    # Save plot to a file
    plt.savefig(plot_path)

    plt.close()


    print(f"Plot saved to {plot_path}")


import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--file', type=str, 
                    help="Path to the .txt file with trainer metrics", required=True)

def main():
    args = parser.parse_args()

    file_path = args.file
    json_path = process_txt_file(file_path)
    train_loss, train_epochs, test_loss, test_epochs = collect_data(json_path)

    # get the plot file path
    plot_dir = os.path.join(os.path.dirname(json_path), 'figs')
    if (not os.path.exists(plot_dir)):
        os.makedirs(plot_dir)
    plot_filename = os.path.splitext(os.path.basename(json_path))[0]
    plot_path = os.path.join(plot_dir, plot_filename+'.png')

    # save the fig to plot_path
    make_plot(train_loss, train_epochs, test_loss, test_epochs, plot_path)

if __name__ == "__main__":
    main()
