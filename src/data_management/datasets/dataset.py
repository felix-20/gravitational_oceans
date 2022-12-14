import os
from random import sample

import h5py
import numpy as np
from torch.utils.data import Dataset

from src.helper.utils import PATH_TO_TRAIN_FOLDER, print_blue, print_red


class GODataset(Dataset):

    def __init__(self, data_folder: str = PATH_TO_TRAIN_FOLDER, transform=None):
        self.transform = transform
        self.frame = []

        for file in os.listdir(f'{data_folder}/no_cw_hdf5'):
            file_data = self._load_data_from_hdf5(f'{data_folder}/no_cw_hdf5/' + file)
            for data in file_data:
                self.frame += [(data, 0)]

        for file in os.listdir(f'{data_folder}/cw_hdf5'):
            file_data = self._load_data_from_hdf5(f'{data_folder}/cw_hdf5/' + file)
            for data in file_data:
                self.frame += [(data, 1)]

    def _load_data_from_hdf5(self, file_path: str) -> dict:
        with h5py.File(file_path, 'r') as hd5_file:
            base_key = list(hd5_file.keys())[0]
            h1_stfts = hd5_file[f'{base_key}/H1/SFTs']
            l1_stfts = hd5_file[f'{base_key}/L1/SFTs']

            print_blue(hd5_file[f'{base_key}/frequency_Hz'][0], '-', hd5_file[f'{base_key}/frequency_Hz'][-1])

            processed_h1_stfts = self._preprocess_stfts(h1_stfts)
            processed_l1_stfts = self._preprocess_stfts(l1_stfts)

            return processed_h1_stfts + processed_l1_stfts

    def _preprocess_stfts(self, stfts: np.array) -> list:
        time_samples = 128
        result = []
        _, timestep_count = stfts.shape
        subset = sample(range(timestep_count), time_samples)
        subset.sort()
        for time_amplitudes in stfts:
            sampled_time_amplitudes = time_amplitudes[subset]
            transformed_sampled_time_amplitudes = np.column_stack((sampled_time_amplitudes.real, sampled_time_amplitudes.imag, np.zeros(time_samples)))
            result += [transformed_sampled_time_amplitudes]

        result = np.transpose(np.array(result), axes=(2, 0, 1))
        assert result.shape == (3, 360, time_samples)
        return [result]

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx) -> dict:
        return self.frame[idx]
