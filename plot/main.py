import json
from typing import Union

from matplotlib import pyplot as plt


def clean_log(path_to_log: str, training: bool = False) -> str:
    """
        Delete all the lines before the last line containing
        "Starting the training"

        :return: the path to the new log file
    """
    with open(path_to_log, 'r') as f:
        lines = f.readlines()
    start_string = 'Starting the training' if training else 'Starting the testing'
    last_start_index = [i for i, line in enumerate(lines) if start_string in line][-1]

    output_lines = lines[last_start_index:]

    new_log_path = path_to_log.replace('.log', '_clean.log')

    with open(new_log_path, 'w') as f:
        f.writelines(output_lines)

    return new_log_path


def extract_data(path_to_log: str, training: bool = False, testing: bool = False) -> Union[dict[str, tuple[list[float], list[float], list[float], list[float]]], dict[str, tuple[list[float], list[float]]]]:
    """
        Extracts the data from the log file where the row start with "History model name:"
        Returns a dictionary with the model name as key and a list of tuples with the data as value.
        The file is in the format
        "History model name: {network_name} History: {'loss': [], 'f1_m':[], 'val_loss': [], val_f1_m': []}"
        We have to return a dictionary containing a list of tuples, each element of the list is a tuple
        Each tuple contains the mean of the values in the list of the history of the model.
    """
    extracted_data = {}
    with open(path_to_log, 'r') as f:
        for line in f:
            if 'History model name:' in line:
                splitted = line.split('History model name: ')
                model_name = splitted[1].split(' History: ')[0]
                history = splitted[1].split(' History: ')[1]
                history = history.replace("'", '"')
                history = json.loads(history)

                if training:
                    if model_name not in extracted_data.keys():
                        extracted_data[model_name] = ([], [], [], [])
                    try:
                        average_loss = sum(history['loss']) / len(history['loss'])
                        average_f1_m = sum(history['f1_m']) / len(history['f1_m'])
                        average_val_loss = sum(history['val_loss']) / len(history['val_loss'])
                        average_val_f1_m = sum(history['val_f1_m']) / len(history['val_f1_m'])
                    except:
                        print("Error in the log file")
                        print(history)
                        continue
                    if average_val_loss > 1 or average_val_f1_m > 1 or average_loss > 1 or average_f1_m > 1:
                        plot = False
                    if plot:
                        extracted_data[model_name][2].append(average_val_loss)
                        extracted_data[model_name][3].append(average_val_f1_m)
                if testing:
                    if model_name not in extracted_data.keys():
                        extracted_data[model_name] = ([], [])
                    average_loss = history[0]
                    average_f1_m = history[1]
                    if average_loss > 1 or average_f1_m > 1:
                        plot = False
                if plot:
                    extracted_data[model_name][0].append(average_loss)
                    extracted_data[model_name][1].append(average_f1_m)

    return extracted_data


def plot_data(extracted_data: Union[dict[str, tuple[list[float], list[float], list[float], list[float]]], dict[str, tuple[list[float], list[float]]]],
              path_to_save: str, show: bool, save: bool):
    """
        Plots the data in the dictionary, each figure is a model
    """
    for model_name, history in extracted_data.items():
        plt.figure()
        plt.title(model_name)
        epochs = [i for i in range(len(history[0]))]
        plt.xlabel("File number")
        plt.ylabel("Value")
        plt.plot(epochs, history[0], label='loss')
        plt.plot(epochs, history[1], label='f1_m')
        folder = 'train' if len(history) == 4 else 'test'
        if len(history) == 4:
            plt.plot(epochs, history[2], label='val_loss')
            plt.plot(epochs, history[3], label='val_f1_m')
        plt.legend()
        if save:
            plt.savefig(path_to_save + folder + "\\" + model_name + '.png')
        if show:
            plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_log', type=str, help='Path to the log file')
    parser.add_argument('path_to_save', type=str, help='Path to save the plot')
    parser.add_argument('--clean', action='store_true', help='Clean the log file')
    parser.add_argument('--show', action='store_true', help='Show the plot')
    parser.add_argument('--save', action='store_true', help='Save the plot')
    parser.add_argument('--train', action='store_true', help='Handle the training log')
    parser.add_argument('--test', action='store_true', help='Handle the testing log')

    args = parser.parse_args()

    new_log_path = args.path_to_log

    if args.clean:
        new_log_path = clean_log(args.path_to_log, args.train)

    data = extract_data(new_log_path, args.train, args.test)
    plot_data(data, args.path_to_save, args.show, args.save)
