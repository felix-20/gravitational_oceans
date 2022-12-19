# 

from collections import defaultdict
import re

from torch.utils.data import Dataset
from timm.data.transforms_factory import create_transform
import torchvision
from torch.utils.data import Dataset
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import timm
import torch
from sklearn.metrics import *
from tqdm import tqdm
import gc
import os
import pandas as pd
import re


POSITIVE_RATE = 0.5
SIGNAL_LOW = 0.02
SIGNAL_HIGH = 0.10
GAUSSIAN_NOISE = 2.
BATCH_SIZE = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'



def get_transforms():
    return torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=0.5, std=0.1)
        ])

class GORealisticNoiseDataset(Dataset):
    def __init__(self, size, df_noise, df_signal, positive_rate=POSITIVE_RATE, is_train=False) -> None:
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

        if self.is_train and GAUSSIAN_NOISE > 0:
            noise = noise + np.random.randn(*noise.shape) * GAUSSIAN_NOISE 

        noise = np.clip(noise, 0, 255).astype(np.uint8)
        return self.transforms(noise)


    def __getitem__(self, index):
        noise_files = self.df_noise.sample().files.values[0]
        
        sig_files = [None, None]
        label = 0
        if np.random.random() < self.positive_rate:
            sig_files = self.df_signal.sample().files.values[0]
            label = 1
        signal_strength = np.random.uniform(SIGNAL_LOW, SIGNAL_HIGH)                    
        return np.concatenate([self.gen_sample(sig, noise, signal_strength) for sig, noise in zip(sig_files, noise_files)], axis=0), label, signal_strength


    def __len__(self):
        return self.size

df_signal = pd.DataFrame(data=[[f] + list(re.findall('.*/(.*)_(.*).png', f)[0]) for f in glob.glob(f'{SIGNAL_DIR}/*')], columns=['name', 'id', 'detector']).sort_values(['id', 'detector'])
df_signal = df_signal.groupby('id').filter(lambda df: len(df) == 2).groupby('id', sort=False).apply(lambda df: df['name'].values).to_frame('files').reset_index()
df_signal_train, df_signal_eval = np.array_split(df_signal, [int(len(df_signal) * 0.9)])

df_noise = pd.DataFrame(data=[[f] + list(re.findall('.*/([^/]*)/([^/]*).png', f)[0]) for f in glob.glob(f'{NOISE_DIR}/*/*.png')], columns=['name', 'id', 'detector']).sort_values(['id', 'detector'])
df_noise = df_noise.groupby('id').filter(lambda df: len(df) == 2).groupby('id', sort=False).apply(lambda df: df['name'].values).to_frame('files').reset_index()
df_noise_train, df_noise_eval = np.array_split(df_noise, [int(len(df_noise) * 0.9)])


ds_eval = GORealisticNoiseDataset(
    len(df_signal_eval), 
    df_noise_eval,
    df_signal_eval
)
dl_eval = torch.utils.data.DataLoader(ds_eval, batch_size=BATCH_SIZE, num_workers=os.cpu_count(), pin_memory=True)