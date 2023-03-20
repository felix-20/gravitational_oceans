# https://www.kaggle.com/code/vslaykovsky/g2net-pytorch-generated-realistic-noise/notebook?scriptVersionId=113484252

import os
from secrets import choice

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from sklearn.metrics import *
from torch.utils.data import Dataset

from src.helper.utils import PATH_TO_CACHE_FOLDER, get_df_dynamic_noise, get_df_signal, print_green, print_red, print_yellow


class GORealisticNoiseDataset(Dataset):
    def __init__(self,
                 size,
                 df_noise,
                 df_signal,
                 positive_rate=0.5,
                 is_train=False,
                 gaussian_noise=3.0,
                 signal_strength_upper=0.1,
                 signal_strength_lower=0.02,
                 artefact_probability=0.3) -> None:
        # df_noise and df_signal are dataframes containing real noise or real signal
        self.df_noise = df_noise
        self.df_signal = df_signal
        self.positive_rate = positive_rate
        self.size = size
        self.transforms = self.get_transforms()
        self.is_train = is_train
        self.gaussian_noise = gaussian_noise
        self.signal_strength_upper = signal_strength_upper
        self.signal_strength_lower = signal_strength_lower
        self.artefact_probability = artefact_probability

    def get_transforms(self):
        return torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=0.5, std=0.1)])

    def vladifier(self, float_image, factor):
        float_image = float_image[:, :4096]
        img = float_image.reshape(360, 256, -1).mean(axis=2)

        return np.clip(img * 255 * factor, 0, 255).astype(np.uint8)
    
    def vertical_line_artefacts(self, combined_image):
        if np.random.random() > self.artefact_probability:
            return combined_image
    
        one_tensor = np.ones_like(combined_image)

        row = np.random.randint(0, combined_image.shape[1])
        factor = np.random.uniform(0, 2)

        one_tensor[row, :] = factor
        return np.multiply(combined_image, one_tensor)
    
    def wave_mover(self, signal):
        column_max = np.max(signal, axis=1)
        zero_cells = np.where(column_max > 10, 1, 0)
        non_zero_indices = np.nonzero(zero_cells)[0]
        first_index = non_zero_indices[0]
        last_index = non_zero_indices[-1]

        # up or down
        if np.random.random() > 0.5:
            offset = np.random.randint(0, first_index)
            signal = np.concatenate([signal[offset:], np.zeros((offset, signal.shape[1]))])
        else:
            offset = np.random.randint(0, signal.shape[0] - last_index)
            offset_inverse = signal.shape[0] - last_index - offset
            signal = np.concatenate([np.zeros((offset_inverse, signal.shape[1])), signal[:last_index+offset]])
        
        return signal

    def gen_sample(self, signal_file, noise_file, signal_strength):
        noise = np.array(cv2.imread(noise_file, cv2.IMREAD_GRAYSCALE))

        noise = self.vladifier(noise / 255.0, 1.0)

        if signal_file:
            signal = np.array(cv2.imread(signal_file, cv2.IMREAD_GRAYSCALE))
            signal = self.vladifier(signal / 255.0, 1.5)
            signal = self.wave_mover(signal)
            noise = noise + signal_strength * signal

        if self.is_train and self.gaussian_noise > 0:
            noise = noise + np.random.randn(*noise.shape) * self.gaussian_noise

        noise = self.vertical_line_artefacts(noise)
        noise = np.clip(noise, 0, 255).astype(np.uint8)
        return self.transforms(noise)


    def __getitem__(self, index):
        noise_files = [choice(self.df_noise), choice(self.df_noise)]

        sig_files = [None, None]
        label = 0
        if np.random.random() < self.positive_rate:
            sig_files = choice(self.df_signal)
            label = 1

        signal_strength = np.random.uniform(self.signal_strength_lower, self.signal_strength_upper)
        img = np.concatenate([self.gen_sample(sig, noise, signal_strength) for sig, noise in zip(sig_files, noise_files)], axis=0)

        # path = f'{PATH_TO_CACHE_FOLDER}/sample{index}_{label}.png'
        # img -= np.min(img)
        # img /= np.max(img)
        # img *= 255
        # cv2.imwrite(path, img[0])

        return img, label, signal_strength


    def __len__(self):
        return self.size*2


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
