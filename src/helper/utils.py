import glob
import json
import os
import re

import h5py
import numpy as np
import pandas as pd


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


if 'KAGGLE_BASE_URL' in os.environ:
    challenge = 'g2net-detecting-continuous-gravitational-waves'
    PATH_TO_TEST_FOLDER = os.path.join('/kaggle', 'input', challenge, 'test')
    PATH_TO_TRAIN_FOLDER = os.path.join('/kaggle', 'input', challenge, 'train')
    PATH_TO_LABEL_FILE = os.path.join('/kaggle', 'input', challenge, 'train_labels.csv')
    PATH_TO_MODEL_FOLDER = os.path.join('/kaggle', 'input', 'models')
    PATH_TO_LOG_FOLDER = os.path.join('/kaggle', 'temp', 'logs')
    PATH_TO_CACHE_FOLDER = os.path.join('/kaggle', 'working', 'cache')
    PATH_TO_SIGNAL_FOLDER = os.path.join('/kaggle', 'working', 'signal')
    PATH_TO_NOISE_FOLDER = os.path.join('/kaggle', 'working', 'noise')
    PATH_TO_DYNAMIC_NOISE_FOLDER = os.path.join(PATH_TO_NOISE_FOLDER, 'dynamic')
    PATH_TO_STATIC_NOISE_FOLDER = os.path.join(PATH_TO_NOISE_FOLDER, 'static')
else:
    PATH_TO_TEST_FOLDER = os.path.join(os.getcwd(), 'test_data')
    PATH_TO_TRAIN_FOLDER = os.path.join(os.getcwd(), 'train_data')
    PATH_TO_MODEL_FOLDER = os.path.join(os.getcwd(), 'models_saved')
    PATH_TO_LOG_FOLDER = os.path.join(os.getcwd(), 'logs')
    PATH_TO_CACHE_FOLDER = os.path.join(os.getcwd(), 'cache')
    PATH_TO_LABEL_FILE = os.path.join(os.getcwd(), 'train_labels.csv')
    PATH_TO_SIGNAL_FOLDER = os.path.join(os.getcwd(), 'signal')
    PATH_TO_NOISE_FOLDER = os.path.join(os.getcwd(), 'noise')
    PATH_TO_DYNAMIC_NOISE_FOLDER = os.path.join(PATH_TO_NOISE_FOLDER, 'dynamic')
    PATH_TO_STATIC_NOISE_FOLDER = os.path.join(PATH_TO_NOISE_FOLDER, 'static')
    PATH_TO_TMP_FOLDER = os.path.join(os.getcwd(), 'tmp')

# setup
if not os.path.isdir(PATH_TO_TRAIN_FOLDER):
    os.makedirs(PATH_TO_TRAIN_FOLDER)
if not os.path.isdir(PATH_TO_TEST_FOLDER):
    os.makedirs(PATH_TO_TEST_FOLDER)
if not os.path.isdir(PATH_TO_MODEL_FOLDER):
    os.makedirs(PATH_TO_MODEL_FOLDER)
if not os.path.isdir(PATH_TO_LOG_FOLDER):
    os.makedirs(PATH_TO_LOG_FOLDER)
if not os.path.isdir(PATH_TO_CACHE_FOLDER):
    os.makedirs(PATH_TO_CACHE_FOLDER)
if not os.path.isdir(PATH_TO_NOISE_FOLDER):
    os.makedirs(PATH_TO_NOISE_FOLDER)
if not os.path.isdir(PATH_TO_SIGNAL_FOLDER):
    os.makedirs(PATH_TO_SIGNAL_FOLDER)
if not os.path.isdir(PATH_TO_DYNAMIC_NOISE_FOLDER):
    os.makedirs(PATH_TO_DYNAMIC_NOISE_FOLDER)
if not os.path.isdir(PATH_TO_STATIC_NOISE_FOLDER):
    os.makedirs(PATH_TO_STATIC_NOISE_FOLDER)
if not os.path.isdir(PATH_TO_TMP_FOLDER):
    os.makedirs(PATH_TO_TMP_FOLDER)

if 'IS_CHARLIE' in os.environ:
    print('We are on Charlie')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def print_red(*text):
    print(f'{bcolors.FAIL}{" ".join([str(t) for t in text])}{bcolors.ENDC}')


def print_blue(*text):
    print(f'{bcolors.OKCYAN}{" ".join([str(t) for t in text])}{bcolors.ENDC}')


def print_green(*text):
    print(f'{bcolors.OKGREEN}{" ".join([str(t) for t in text])}{bcolors.ENDC}')


def print_yellow(*text):
    print(f'{bcolors.WARNING}{" ".join([str(t) for t in text])}{bcolors.ENDC}')


def open_hdf5_file(path_to_file):
    result = {}
    with h5py.File(path_to_file, 'r') as hd5_file:
        base_key = list(hd5_file.keys())[0]
        result['base_key'] = base_key
        result['frequencies'] = np.array(hd5_file[f'{base_key}/frequency_Hz'])
        result['h1'] = {}
        result['l1'] = {}
        result['h1']['amplitudes'] = np.array(hd5_file[f'{base_key}/H1/SFTs'])
        result['l1']['amplitudes'] = np.array(hd5_file[f'{base_key}/L1/SFTs'])
        result['h1']['timestamps'] = np.array(hd5_file[f'{base_key}/H1/timestamps_GPS'])
        result['l1']['timestamps'] = np.array(hd5_file[f'{base_key}/L1/timestamps_GPS'])

    return result


def get_df_dynamic_noise() -> pd.DataFrame:
    assert len(os.listdir(PATH_TO_DYNAMIC_NOISE_FOLDER)) != 0, 'There must be data in noise folder'
    return [os.path.join(PATH_TO_DYNAMIC_NOISE_FOLDER, p) for p in os.listdir(PATH_TO_DYNAMIC_NOISE_FOLDER)]


def get_df_static_noise() -> pd.DataFrame:
    assert len(os.listdir(PATH_TO_STATIC_NOISE_FOLDER)) != 0, 'There must be data in static_noise folder'
    return [os.path.join(PATH_TO_STATIC_NOISE_FOLDER, p) for p in os.listdir(PATH_TO_STATIC_NOISE_FOLDER)]


def get_df_signal() -> pd.DataFrame:
    assert len(os.listdir(PATH_TO_SIGNAL_FOLDER)) != 0, 'There must be data in signal folder'
    all_files = [os.path.join(PATH_TO_SIGNAL_FOLDER, p) for p in os.listdir(PATH_TO_SIGNAL_FOLDER)]
    all_files = sorted(all_files)
    offset = len(all_files) // 2
    return [(all_files[i], all_files[i+offset]) for i in range(offset)]


if __name__ == '__main__':
    print_red('This', 'text', 'is red', 1, 23)
    print_blue('This', 'text', 'is blue', 1, 23)
    print_green('This', 'text', 'is green', 1, 23)
    print_yellow('This', 'text', 'is yellow', 1, 23)
