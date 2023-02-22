# https://www.kaggle.com/code/vslaykovsky/g2net-pytorch-generated-realistic-noise/notebook?scriptVersionId=113484252

import os

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from sklearn.metrics import *
from torch.utils.data import Dataset
from secrets import choice

from src.helper.utils import get_df_dynamic_noise, get_df_signal, print_red, PATH_TO_CACHE_FOLDER


class GORealisticNoiseDataset(Dataset):
    def __init__(self,
                 size,
                 df_noise,
                 df_signal,
                 signal_strength=1.0,
                 positive_rate=0.5,
                 is_train=False,
                 gaussian_noise=1.0) -> None:
        # df_noise and df_signal are dataframes containing real noise or real signal
        self.df_noise = df_noise
        self.df_signal = df_signal
        self.positive_rate = positive_rate
        self.size = size
        self.transforms = self.get_transforms()
        self.is_train = is_train
        self.gaussian_noise = gaussian_noise
        self.signal_strength = signal_strength

    def get_transforms(self):
        return torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=0.5, std=0.1)])

    def gen_sample(self, signal, noise, signal_strength):
        noise = np.array(Image.open(noise))
        noise = noise[:,:4500]

        if True: #signal:
            signal = np.array(Image.open(signal))
            signal = signal[:,:4500]

            #noise = noise + signal_strength * signal

        if self.is_train and self.gaussian_noise > 0:
            noise = noise + np.random.randn(*noise.shape) * self.gaussian_noise

        noise = np.clip(noise, 0, 255).astype(np.uint8)
        return self.transforms(noise)


    def __getitem__(self, index):
        noise_files = [choice(self.df_noise), choice(self.df_noise)]

        sig_files = [None, None]
        label = 0
        if np.random.random() < self.positive_rate:
            sig_files = choice(self.df_signal)
            label = 1
        signal_strength = self.signal_strength # np.random.uniform(0.02, 0.10)
        img = np.concatenate([self.gen_sample(sig, noise, signal_strength) for sig, noise in zip(sig_files, noise_files)], axis=0)
        path = os.path.join(PATH_TO_CACHE_FOLDER, 'img', f'{index}.jpg')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print_red(img[0].shape)
        img += abs(np.min(img))
        img /= np.max(img)
        img *= 255
        print_red(img[0])
        cv2.imwrite(f'{index}.jpg', img[0])
        #i.save(path)
        print(f'Done {path}')
        # exit(0)
        return img, label, signal_strength


    def __len__(self):
        return self.size


if __name__ == '__main__':
    batch_size = 32

    df_signal = get_df_signal()
    df_signal_train, df_signal_eval = np.array_split(df_signal, [int(len(df_signal) * 0.9)])

    df_noise = get_df_dynamic_noise()
    df_noise_train, df_noise_eval = np.array_split(df_noise, [int(len(df_noise) * 0.9)])


    ds_eval = GORealisticNoiseDataset(
        len(df_signal_eval),
        df_noise_eval,
        df_signal_eval
    )
    dl_eval = torch.utils.data.DataLoader(ds_eval, batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
