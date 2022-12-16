from os import listdir, makedirs, path
from random import shuffle

import numpy as np
import torch
from tqdm import tqdm
import gzip
from random import randint

from src.ai_nets.pretrained_efficientnet import dataload, normalize, preprocess
from src.data_management.dataset import GODataset
from src.helper.utils import PATH_TO_CACHE_FOLDER, print_green, print_blue, print_red, print_yellow
from src.data_management.hdf5_sorter import GOHDF5Sorter


class GOBetterCRNNDataset(GODataset):

    def __init__(self, data_folder: str = path.join(PATH_TO_CACHE_FOLDER, 'pre_predicted'), sequence_length: int = 5):
        print_green('Dataset ready!')

        self.data_folder = data_folder
        self.sequence_length = sequence_length

        # npy-files are prepredicted by the best possible CNN
        self.npy_files = listdir(data_folder)
        self.sorter = GOHDF5Sorter('train_labels.csv')

        self.sorter.get_sorted_frequencies()

        self.last_accessed_files = []

        print_blue(self.__getitem__(0)[0].shape)
        self.reset_last_accessed_files()

    def __len__(self) -> int:
        return len(self.npy_files) - self.sequence_length + 1

    def __getitem__(self, idx) -> dict:
        files_with_labels = self.sorter.get_sorted_frequencies()[idx : idx+self.sequence_length]
        self.last_accessed_files += [[file_path for file_path, _ in files_with_labels]]

        cnn_predictions = []
        labels = []
        for hdf5_file_path, label in files_with_labels:
            gz_npy_file_path = path.join(self.data_folder, path.basename(hdf5_file_path)[:-5] + '.npy.gz')
            with gzip.open(gz_npy_file_path, 'rb') as npy_file:
                data = np.load(npy_file)
                cnn_predictions += [data[randint(0, len(data) - 1)]]
            labels += [int(label)]
        
        result_tensor = torch.tensor(np.concatenate(cnn_predictions, axis=2))
        
        return (result_tensor, torch.tensor(labels, dtype=torch.int32))
    
    def get_last_accessed_files(self):
        return self.last_accessed_files

    def reset_last_accessed_files(self):
        self.last_accessed_files = []

if __name__ == '__main__':
    item = GOBetterCRNNDataset().__getitem__(0)
    print(item[0].shape)
    print(item)
