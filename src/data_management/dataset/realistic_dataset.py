# https://www.kaggle.com/code/vslaykovsky/g2net-pytorch-generated-realistic-noise/notebook?scriptVersionId=113484252

import re
import torchvision
from torch.utils.data import Dataset
import glob
import numpy as np
from PIL import Image
import torch
from sklearn.metrics import *
import os
import pandas as pd

from src.helper.utils import get_df_noise, get_df_signal


def get_transforms():
    return torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=0.5, std=0.1)
        ])

class GORealisticNoiseDataset(Dataset):
    def __init__(self, size, df_noise, df_signal, positive_rate=0.5, is_train=False) -> None:
        # df_noise and df_signal are dataframes containing real noise or real signal
        self.df_noise = df_noise
        self.df_signal = df_signal
        self.positive_rate = positive_rate
        self.size = size
        self.transforms = get_transforms()
        self.is_train = is_train


    def gen_sample(self, signal, noise, signal_strength):
        # print(signal, noise)
        noise = np.array(Image.open(noise))
        # print(np.mean(noise.flatten() / 255), np.std(noise.flatten()/ 255))
        if signal:
            signal = np.array(Image.open(signal))
            noise = noise + signal_strength * signal

        if self.is_train and 2.0 > 0:
            noise = noise + np.random.randn(*noise.shape) * 2.0 

        noise = np.clip(noise, 0, 255).astype(np.uint8)
        return self.transforms(noise)


    def __getitem__(self, index):
        noise_files = self.df_noise.sample().files.values[0]
        
        sig_files = [None, None]
        label = 0
        if np.random.random() < self.positive_rate:
            sig_files = self.df_signal.sample().files.values[0]
            label = 1
        signal_strength = np.random.uniform(0.02, 0.10)                    
        return np.concatenate([self.gen_sample(sig, noise, signal_strength) for sig, noise in zip(sig_files, noise_files)], axis=0), label, signal_strength


    def __len__(self):
        return self.size


if __name__ == '__main__':
    batch_size = 32

    df_signal = get_df_signal()
    df_signal_train, df_signal_eval = np.array_split(df_signal, [int(len(df_signal) * 0.9)])

    df_noise = get_df_noise()
    df_noise_train, df_noise_eval = np.array_split(df_noise, [int(len(df_noise) * 0.9)])


    ds_eval = GORealisticNoiseDataset(
        len(df_signal_eval), 
        df_noise_eval,
        df_signal_eval
    )
    dl_eval = torch.utils.data.DataLoader(ds_eval, batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)