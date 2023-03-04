import h5py
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.helper.utils import PATH_TO_TRAIN_FOLDER, PATH_TO_CACHE_FOLDER


def open_hdf5_file(path_to_file: str):
    result = {}
    with h5py.File(path_to_file, 'r') as hd5_file:
        base_key = list(hd5_file.keys())[0]
        result['base_key'] = base_key
        result['frequencies'] = np.array(hd5_file[f'{base_key}/frequency_Hz'])
        result['h1'] = {}
        result['l1'] = {}
        result['h1']['amplitudes'] = np.array(hd5_file[f'{base_key}/H1/SFTs'])
        result['l1']['amplitudes'] = np.array(hd5_file[f'{base_key}/L1/SFTs'])
        result['h1']['timestamps'] = np.array(hd5_file[f'{base_key}/H1/timestamps_GPS'])
        result['l1']['timestamps'] = np.array(hd5_file[f'{base_key}/L1/timestamps_GPS'])

    return result


real_h1_means = []
real_h1_std = []
imag_h1_means = []
imag_h1_std = []
real_l1_means = []
real_l1_std = []
imag_l1_means = []
imag_l1_std = []

def compute_noise_statistics(files):
    global real_h1_means,\
        real_h1_std,\
        imag_h1_means,\
        imag_h1_std,\
        real_l1_means,\
        real_l1_std,\
        imag_l1_means,\
        imag_l1_std
        
        
    for file_path in tqdm(files):
        data = open_hdf5_file(file_path)
        real_amplitudes_h1 = np.real(data['h1']['amplitudes'])
        imag_amplitudes_h1 = np.imag(data['h1']['amplitudes'])
        real_amplitudes_l1 = np.real(data['l1']['amplitudes'])
        imag_amplitudes_l1 = np.imag(data['l1']['amplitudes'])


        real_h1_means += [np.mean(real_amplitudes_h1)]
        real_h1_std += [np.std(real_amplitudes_h1)]
        imag_h1_means += [np.mean(imag_amplitudes_h1)]
        imag_h1_std += [np.std(imag_amplitudes_h1)]

        real_l1_means += [np.mean(real_amplitudes_l1)]
        real_l1_std += [np.std(real_amplitudes_l1)]
        imag_l1_means += [np.mean(imag_amplitudes_l1)]
        imag_l1_std += [np.std(imag_amplitudes_l1)]


def visualize_noise_statistics():
    fig, axs = plt.subplots(2, 4)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    axs[0,0].hist(real_h1_means)
    axs[0,0].set_title('real_h1_means')
    axs[0,1].hist(real_h1_std)
    axs[0,1].set_title('real_h1_std')
    axs[0,2].hist(imag_h1_means)
    axs[0,2].set_title('imag_h1_means')
    axs[0,3].hist(imag_h1_std)
    axs[0,3].set_title('imag_h1_std')
    axs[1,0].hist(real_l1_means)
    axs[1,0].set_title('real_l1_means')
    axs[1,1].hist(real_l1_std)
    axs[1,1].set_title('real_l1_std')
    axs[1,2].hist(imag_l1_means)
    axs[1,2].set_title('imag_l1_means')
    axs[1,3].hist(imag_l1_std)
    axs[1,3].set_title('imag_l1_std')
    fig.savefig(os.path.join(PATH_TO_CACHE_FOLDER, 'statistics.png'))

def print_noise_statistics():
    print(f'mean real_h1_means: {np.mean(real_h1_means)}')
    print(f'mean real_h1_std: {np.mean(real_h1_std)}')
    print(f'mean imag_h1_means: {np.mean(imag_h1_means)}')
    print(f'mean imag_h1_std: {np.mean(imag_h1_std)}')
    print(f'mean real_l1_means: {np.mean(real_l1_means)}')
    print(f'mean real_l1_std: {np.mean(real_l1_std)}')
    print(f'mean imag_l1_means: {np.mean(imag_l1_means)}')
    print(f'mean imag_l1_std: {np.mean(imag_l1_std)}')

    print('-------------------')

    print(f'median real_h1_means: {np.median(real_h1_means)}')
    print(f'median real_h1_std: {np.median(real_h1_std)}')
    print(f'median imag_h1_means: {np.median(imag_h1_means)}')
    print(f'median imag_h1_std: {np.median(imag_h1_std)}')
    print(f'median real_l1_means: {np.median(real_l1_means)}')
    print(f'median real_l1_std: {np.median(real_l1_std)}')
    print(f'median imag_l1_means: {np.median(imag_l1_means)}')
    print(f'median imag_l1_std: {np.median(imag_l1_std)}')



if __name__=='__main__':
    all_hdf5_files = list(filter(lambda filename: filename.endswith('.hdf5'), os.listdir(os.path.join(PATH_TO_TRAIN_FOLDER, 'no_cw_hdf5'))))
    all_files_complete_path = [os.path.join(PATH_TO_TRAIN_FOLDER, 'no_cw_hdf5', filename) for filename in all_hdf5_files]
    assert all_files_complete_path, f'There are no files, that can be analysed in {PATH_TO_TRAIN_FOLDER}'
    compute_noise_statistics(all_files_complete_path)
    visualize_noise_statistics()
    print_noise_statistics()
