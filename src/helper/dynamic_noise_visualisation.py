import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

from src.data_management.datasets.realistic_dataset import GORealisticNoiseDataset
from src.helper.utils import PATH_TO_DYNAMIC_NOISE_FOLDER, PATH_TO_CACHE_FOLDER, PATH_TO_SIGNAL_FOLDER, print_yellow, print_blue, print_green, print_red, normalize_image


def open_image(file):
    return cv2.imread(file)


def build_graph(noise, name):
    means = np.mean(noise, axis=0)
    print_yellow(means.shape)
    x = range(len(means))
    plt.plot(x, means)
    plt.savefig(os.path.join(PATH_TO_CACHE_FOLDER, 'statistics_'+name+'.png'))
    plt.clf()


def get_noise_signal_samples():
    # get noise and signal samples
    all_noise_files = [os.path.join(PATH_TO_DYNAMIC_NOISE_FOLDER, f) for f in os.listdir(PATH_TO_DYNAMIC_NOISE_FOLDER)]
    # noise = (np.random.choice(all_noise_files), np.random.choice(all_noise_files))
    n = np.random.choice(all_noise_files)
    noise = (n, n)
    all_signal_files = list(filter(lambda x: x.startswith('H1'), os.listdir(PATH_TO_SIGNAL_FOLDER)))
    h1_signal = os.path.join(PATH_TO_SIGNAL_FOLDER, np.random.choice(all_signal_files))
    l1_signal = os.path.join(os.path.dirname(h1_signal), os.path.basename(h1_signal).replace('H1', 'L1'))
    signal = (h1_signal, l1_signal)

    return noise, signal


def get_one_sample(signal_strength: float=0.17, 
                   path_to_noise_files: tuple = None, 
                   path_to_signal_files: tuple = None, 
                   should_normalize: bool=True):
    
    if path_to_noise_files and path_to_signal_files:
        # if noise and signal paths are provided, use it
        signal = path_to_signal_files
        noise = path_to_noise_files
    else:
        # if not generate own samples
        noise, signal = get_noise_signal_samples()
    
    dataset = GORealisticNoiseDataset(0,0,0)
    img = np.concatenate([dataset.gen_sample(sig, noise, signal_strength) for sig, noise in zip(signal, noise)], axis=0)

    if should_normalize:
        return normalize_image(img)
    else:
        return img


def visualize_signal_strengths(path_to_noise_files, path_to_signal_files, signal_strengths):
    for s in signal_strengths:
        path = os.path.join(PATH_TO_CACHE_FOLDER, 'signal_strength_visualization')
        os.makedirs(os.path.dirname(path), exist_ok=True)

        img = get_one_sample(s, path_to_noise_files, path_to_signal_files)

        cv2.imwrite(os.path.join(path, f'img_{s}_v0.png'), img[0])
        cv2.imwrite(os.path.join(path, f'img_{s}_v1.png'), img[1])


if __name__=='__main__':
    # path = PATH_TO_DYNAMIC_NOISE_FOLDER
    # all_files_complete_path = [os.path.join(path, filename) for filename in os.listdir(path)]
    
    # for noise_file in all_files_complete_path:
    #     data = open_image(noise_file)
    #     build_graph(data, os.path.basename(noise_file)[:-4])
    
    noise, signal = get_noise_signal_samples()
    visualize_signal_strengths(noise, signal, [0.0, 1.0])
