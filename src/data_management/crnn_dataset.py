import torch
import numpy as np

from src.helper.utils import PATH_TO_TRAIN_FOLDER
from src.data_management.dataset import GODataset

class GOCRNNDataset(GODataset):

    def __init__(self, data_folder: str = PATH_TO_TRAIN_FOLDER, transform=None):
        super().__init__(data_folder, transform)

        self.frame = [item for item in self.frame for _ in range(30)]

    def __getitem__(self, idx) -> dict:
        return (np.transpose(self.frame[idx][0], (0, 2, 1)), torch.tensor([self.frame[idx][1]], dtype=torch.int32))


