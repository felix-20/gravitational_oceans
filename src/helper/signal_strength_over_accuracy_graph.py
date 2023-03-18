from os import path

import numpy as np
from matplotlib import pyplot as plt

from src.helper.utils import PATH_TO_CACHE_FOLDER

if __name__ == '__main__':
    input_path = path.join(PATH_TO_CACHE_FOLDER, 'signal_strength_results', 'signal_strengths_paper.csv')
    output_path = path.join(PATH_TO_CACHE_FOLDER, 'signal_strength_results', 'signal_strengths_paper.png')

    lines = sum(1 for _ in open(input_path))

    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0, 1, lines)]

    fig = plt.figure()
    ax = plt.subplot(111)

    handles = []
    line_index = 0
    all_data = []
    with open(input_path, 'r') as file:
        for line in file:
            data = line.split(',')
            data = [float(x) for x in data]
            all_data += [data]

    all_data = sorted(all_data, key=lambda x: x[0])

    for line in all_data:
        signal_strength = line[0]
        max_accuracy = line[1]
        accuracies = line[2:]
        epochs = len(accuracies)

        handle = ax.plot(range(epochs), accuracies, label=f'{signal_strength:.5f}', color=colors[line_index])
        handles += handle
        line_index += 1

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.set_title('Accuracy as a function of signal strengths')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')

    plt.legend(handles=handles[::-1], loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(output_path)
