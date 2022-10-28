import os

import h5py
import numpy as np
from torch.utils.data import Dataset


class GODataset(Dataset):

    def __init__(self, data_folder: str, transform=None):
        self.transform = transform
        self.frame = []

        for file in os.listdir(f'{data_folder}/no_cw_hdf5'):
            file_data = self._load_data_from_hdf5('./data/no_cw_hdf5/' + file)
            self.frame += [{'signal': file_data['H1'], 'label': 0}]
            self.frame += [{'signal': file_data['L1'], 'label': 0}]

        for file in os.listdir(f'{data_folder}/cw_hdf5'):
            file_data = self._load_data_from_hdf5('./data/cw_hdf5/' + file)
            self.frame += [{'signal': file_data['H1'], 'label': 1}]
            self.frame += [{'signal': file_data['L1'], 'label': 1}]

    def _load_data_from_hdf5(self, file_path: str) -> dict:
        with h5py.File(file_path, 'r') as hd5_file:
            base_key = list(hd5_file.keys())[0]
            h1_stfts = hd5_file[f'{base_key}/H1/SFTs'][0]
            l1_stfts = hd5_file[f'{base_key}/L1/SFTs'][0]

            processed_h1_stfts = self._preprocess_stfts(h1_stfts)
            processed_l1_stfts = self._preprocess_stfts(l1_stfts)

            return {'H1': processed_h1_stfts, 'L1': processed_l1_stfts}

    def _preprocess_stfts(self, stfts: np.array) -> np.array:
        print(stfts)
        time_samples = 512
        result = np.array([])
        for time_amplitudes in stfts:
            sampled_time_amplitudes = time_amplitudes[np.random.choice(len(time_amplitudes), size=time_samples, replace=False)]
            transformed_sampled_time_amplitudes = sampled_time_amplitudes.view('(2,)float')
            np.concatenate(result, transformed_sampled_time_amplitudes)

        assert result.shape == (360, time_samples, 2)
        return result

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx) -> dict:
        return self.frame[idx]
