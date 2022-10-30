import os

import h5py
import numpy as np
from torch.utils.data import Dataset

from utils import print_red


class GODataset(Dataset):

    def __init__(self, data_folder: str, transform=None):
        self.transform = transform
        self.frame = []

        for file in os.listdir(f'{data_folder}/no_cw_hdf5'):
            file_data = self._load_data_from_hdf5('./data/no_cw_hdf5/' + file)
            self.frame += [(file_data['H1'], 0)]
            self.frame += [(file_data['L1'], 0)]

        for file in os.listdir(f'{data_folder}/cw_hdf5'):
            file_data = self._load_data_from_hdf5('./data/cw_hdf5/' + file)
            self.frame += [(file_data['H1'], 1)]
            self.frame += [(file_data['L1'], 1)]

    def _load_data_from_hdf5(self, file_path: str) -> dict:
        with h5py.File(file_path, 'r') as hd5_file:
            base_key = list(hd5_file.keys())[0]
            h1_stfts = hd5_file[f'{base_key}/H1/SFTs']
            l1_stfts = hd5_file[f'{base_key}/L1/SFTs']

            processed_h1_stfts = self._preprocess_stfts(h1_stfts)
            processed_l1_stfts = self._preprocess_stfts(l1_stfts)

            return {'H1': processed_h1_stfts, 'L1': processed_l1_stfts}

    def _preprocess_stfts(self, stfts: np.array) -> np.array:
        time_samples = 512
        result = []
        for time_amplitudes in stfts:
            sampled_time_amplitudes = time_amplitudes[np.random.choice(len(time_amplitudes), size=time_samples, replace=False)]
            transformed_sampled_time_amplitudes = np.column_stack((sampled_time_amplitudes.real, sampled_time_amplitudes.imag))
            result += [transformed_sampled_time_amplitudes]

        result = np.array(result)
        assert result.shape == (360, time_samples, 2)
        return result

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx) -> dict:
        stft_data = self.frame[idx][0]
        transformed_stft_data = np.transpose(stft_data, axes=(2, 0, 1))
        print_red('in dataset.py->GODataset->__get_item__ after transform', transformed_stft_data.shape)
        return (transformed_stft_data, self.frame[idx][1])
