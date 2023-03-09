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
import cv2

from src.helper.utils import get_df_dynamic_noise, get_df_signal, print_red, print_yellow, print_green


class GORealisticNoiseDataset(Dataset):
    def __init__(self,
                 size,
                 df_noise,
                 df_signal,
                 positive_rate=0.5,
                 is_train=False,
                 gaussian_noise=1.0,
                 signal_strength=0.1) -> None:
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
        noise = noise[:,:2000]

        if signal:
            signal = np.array(Image.open(signal))
            signal = signal[:,:2000]

            noise = noise + signal_strength * signal

        #if self.is_train and self.gaussian_noise > 0:
        #    noise = noise + np.random.randn(*noise.shape) * self.gaussian_noise

        noise = np.clip(noise, 0, 255).astype(np.uint8)
        return self.transforms(noise)


    def __getitem__(self, index):
        noise_files = [choice(self.df_noise), choice(self.df_noise)]

        sig_files = [None, None]
        label = 0
        if np.random.random() < self.positive_rate:
            sig_files = choice(self.df_signal)
            label = 1

        img = np.concatenate([self.gen_sample(sig, noise, self.signal_strength) for sig, noise in zip(sig_files, noise_files)], axis=0)

        # path = f'./gravitational_oceans/tmp/sample{index}_{label}.png'
        # print_red(img[0].shape)
        # img += abs(np.min(img))
        # img /= np.max(img)
        # img *= 255
        # cv2.imwrite(path, img[0])
        # print_green(f'{index} done is signal: {label==1}')
        # exit(1)

        return img, label, self.signal_strength


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
