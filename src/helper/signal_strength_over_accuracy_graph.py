import numpy as np

from matplotlib import pyplot as plt
from os import path

from src.helper.utils import PATH_TO_CACHE_FOLDER

if __name__ == '__main__':
    input_path = path.join(PATH_TO_CACHE_FOLDER, 'signal_strength_over_accuracy.csv')
    output_path = path.join(PATH_TO_CACHE_FOLDER, 'signal_strength_over_accuracy.png')

    lines = sum(1 for _ in open(input_path))

    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0, 1, lines)]

    handles = []
    line_index = 0
    with open(input_path, 'r') as file:
        for line in file:
            data = line.split(',')
            data = [float(x) for x in data]

            signal_strength = data[0]
            max_accuracy = data[1]
            accuracies = data[2:]
            epochs = len(accuracies)
    
            handle = plt.plot(range(epochs), accuracies, label=f'{signal_strength:.2f}', color=colors[line_index])
            handles += handle
            line_index += 1
    
    plt.legend(handles=handles[::-1])
    plt.savefig(output_path)