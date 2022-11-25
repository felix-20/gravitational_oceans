from os import listdir, makedirs, path
from random import shuffle

import numpy as np
import torch
from tqdm import tqdm

from src.data_management.dataset import GODataset
from src.helper.utils import PATH_TO_TEST_FOLDER, print_green
from src.ai_nets.pretrained_efficientnet import normalize, dataload, preprocess


class GOBetterCRNNDataset(GODataset):

    def __init__(self, data_folder: str = PATH_TO_TEST_FOLDER, sequence_length: int = 5):
        print_green('Dataset ready!')

        no_cw_folder = path.join(data_folder, 'no_cw_hdf5')
        cw_folder = path.join(data_folder, 'cw_hdf5')

        self.frame = []
        self.frame += [(path.join(no_cw_folder, file_name), 0) for file_name in listdir(no_cw_folder)]
        self.frame += [(path.join(cw_folder, file_name), 1) for file_name in listdir(cw_folder)]

        shuffle(self.frame)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx) -> dict:
        file_path, label = self.frame[idx]

        _, input, H1, L1 = dataload(file_path)
        tta = preprocess(64, input, H1, L1)
        return (tta, label)

if __name__ == '__main__':
    item = GOBetterCRNNDataset().__getitem__(0)
    print(item[0].shape)
    print(item)
