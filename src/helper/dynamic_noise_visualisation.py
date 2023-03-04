import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.helper.utils import PATH_TO_DYNAMIC_NOISE_FOLDER, PATH_TO_CACHE_FOLDER, print_yellow, print_blue, print_green, print_red


def open_image(file):
    return cv2.imread(file)

def build_graph(noise, name):

    means = np.mean(noise, axis=0)

    print_yellow(means.shape)

    x = range(len(means))
    plt.plot(x, means)
    plt.savefig(os.path.join(PATH_TO_CACHE_FOLDER, 'statistics_'+name+'.png'))
    plt.clf()



if __name__=='__main__':
    path = PATH_TO_DYNAMIC_NOISE_FOLDER
    all_files_complete_path = [os.path.join(path, filename) for filename in os.listdir(path)]
    
    for noise_file in all_files_complete_path:
        data = open_image(noise_file)
        build_graph(data, os.path.basename(noise_file)[:-4])

