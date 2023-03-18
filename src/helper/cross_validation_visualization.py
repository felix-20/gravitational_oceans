import json
from os import listdir, path

import matplotlib.pyplot as plt
import numpy as np

from src.helper.utils import PATH_TO_CACHE_FOLDER

PATH_TO_JSON_FILES = path.join(PATH_TO_CACHE_FOLDER, 'cross_validation')


def visualize_for_one_ratio(accuracy_dict: dict, ratio: float):
    plt.clf()
    ax = plt.gca()
    ax.set_ylim([0.7, 1.03])
    for n, accs in accuracy_dict.items():
        plt.plot(range(len(accs)), accs, label=n)
    plt.title(f'Training - Evaluation Ratio {ratio}')
    plt.legend()
    plt.savefig(path.join(PATH_TO_JSON_FILES, f'{ratio}.png'))


if __name__ == '__main__':
    all_files = [path.join(PATH_TO_JSON_FILES, f) for f in listdir(PATH_TO_JSON_FILES)]

    final_dict = {}
    for f in all_files:
        with open(f, 'r') as json_file:
            result_dict = json.load(json_file)
        visualize_for_one_ratio(result_dict['accs'], path.basename(f)[:-5])
