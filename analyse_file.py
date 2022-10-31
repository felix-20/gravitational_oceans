import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from utils import print_blue


def allkeys(obj):
    'Recursively find all keys in an h5py.Group.'
    keys = (obj.name,)
    if isinstance(obj, h5py.Group):
        for key, value in obj.items():
            if isinstance(value, h5py.Group):
                keys = keys + allkeys(value)
            else:
                keys = keys + (value.name,)
    return keys

file_path = 'data/cw_hdf5/signal0_0.hdf5'
#file_path = './example/00054c878.hdf5'

def analyse_keys():
    with h5py.File(file_path, 'r') as file:
        all_keys = allkeys(file)
        print('\n'.join(all_keys))

        print('------------')

        for key in all_keys:
            obj = file[key]
            if isinstance(obj, h5py.Dataset):
                print(f'{key} -> {obj.shape}')


def visualize_data(path_to_file: str):
    with h5py.File(path_to_file, 'r') as hd5_file:
        base_key = list(hd5_file.keys())[0]
        amplitudes = np.array(hd5_file[f'{base_key}/H1/SFTs'])
        frequency = hd5_file[f'{base_key}/frequency_Hz']
        timestamps = hd5_file[f'{base_key}/H1/timestamps_GPS']
        plot_real_imag_spectrograms(timestamps, frequency, amplitudes)

def plot_real_imag_spectrograms(timestamps, frequency, fourier_data, name='tested_img'):
    'uses simple plt.pcolormesh'
    fig, axs = plt.subplots(1, 2, figsize=(16, 10))

    for ax in axs:
        ax.set(xlabel='SFT index', ylabel='Frequency [Hz]')

    time_in_days = (timestamps - timestamps[0]) / 1800

    axs[0].set_title('SFT Real part')
    c = axs[0].pcolormesh(
        time_in_days,
        frequency,
        fourier_data.real,
        norm=colors.CenteredNorm(),
    )
    fig.colorbar(c, ax=axs[0], orientation='horizontal', label='Fourier Amplitude')

    axs[1].set_title('SFT Imaginary part')
    c = axs[1].pcolormesh(
        time_in_days,
        frequency,
        fourier_data.imag,
        norm=colors.CenteredNorm(),
    )

    fig.colorbar(c, ax=axs[1], orientation='horizontal', label='Fourier Amplitude')

    path_to_images = './data/visualize/'
    if not os.path.isdir(path_to_images):
        os.makedirs(path_to_images)
    plt.savefig(f'{path_to_images}{name}')
    return fig, axs

visualize_data('./data/cw_hdf5/signal0_0.hdf5')
