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
        train_loss=[]
        train_epochs=[]
        train_accuracy=[]
        test_loss=[]
        test_epochs=[]
        test_accuracy=[]
        for data in json_data:
            if 'eval_train_accuracy' in data:
                train_loss.append(data['eval_train_loss'])
                train_accuracy.append(data['eval_train_accuracy'])
                train_epochs.append(data['epoch'])
            if 'eval_test_accuracy' in data:
                test_loss.append(data['eval_test_loss'])
                test_accuracy.append(data['eval_test_accuracy'])
                test_epochs.append(data['epoch'])
        
        return train_loss, train_accuracy, train_epochs, test_loss, test_accuracy, test_epochs
    
    json_data = read_json_file(file_path)
    train_loss, train_accuracy, train_epochs, test_loss, test_accuracy, test_epochs = collect_accuracy(json_data)

    return train_loss, train_accuracy, train_epochs, test_loss, test_accuracy, test_epochs

def make_plot(file_path):

    json_path = process_txt_file(file_path)
    train_loss, train_accuracy, train_epochs, test_loss, test_accuracy, test_epochs = collect_data(json_path)

    # get the plot file path
    plot_dir = os.path.join(os.path.dirname(json_path), 'figs')
    if (not os.path.exists(plot_dir)):
        os.makedirs(plot_dir)
    plot_filename = os.path.splitext(os.path.basename(json_path))[0]
    plot_path = os.path.join(plot_dir, plot_filename+'.png')
    
    # Create a figure and a set of subplots
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # First subplot: Accuracy vs Epochs
    ax[0].plot(train_epochs, train_loss, 'r-', label='train loss')
    ax[0].plot(test_epochs, test_loss, 'b-', label='test loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Loss vs Epochs')
    ax[0].legend()

    # Second subplot: Loss vs Epochs
    ax[1].plot(train_epochs, train_accuracy, 'r-', label='train ccuracy')
    ax[1].plot(test_epochs, test_accuracy, 'b-', label='train ccuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title('Accuracy vs Epochs')
    ax[1].legend()

    title = os.path.splitext(os.path.basename(file_path))[0]
    fig.suptitle(title, fontsize=16)

    # Save plot to a file
    plt.savefig(plot_path)

    plt.close()


    print(f"Plot saved to {plot_path}")


import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--file', type=str, 
                    help="Path to the .txt file with trainer metrics", default=None)

parser.add_argument('--folder', type=str,
                    help="Path to logs folder", default=None)

def main():
    args = parser.parse_args()

    file_path = args.file
    if(file_path is not None):
        # save the fig to plot_path
        make_plot(file_path)
    else:
        folder_path = args.folder
        if(folder_path is not None):
            for filename in os.listdir(folder_path):
                if filename.endswith(".txt"):
                    file_path = os.path.join(folder_path, filename)
                    make_plot(file_path)
        else:
            raise FileNotFoundError("Please provide file or folder path")


if __name__ == "__main__":
    main()
