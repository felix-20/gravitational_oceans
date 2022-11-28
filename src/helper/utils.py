import os, h5py
import numpy as np

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

PATH_TO_TEST_FOLDER = os.path.join(os.getcwd(), 'test_data')
PATH_TO_TRAIN_FOLDER = os.path.join(os.getcwd(), 'train_data')
PATH_TO_MODEL_FOLDER = os.path.join(os.getcwd(), 'models_saved')
PATH_TO_LOG_FOLDER = os.path.join(os.getcwd(), 'logs')
PATH_TO_CACHE_FOLDER = os.path.join(os.getcwd(), 'cache')

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


if __name__ == '__main__':
    print_red('This', 'text', 'is red', 1, 23)
    print_blue('This', 'text', 'is blue', 1, 23)
    print_green('This', 'text', 'is green', 1, 23)
    print_yellow('This', 'text', 'is yellow', 1, 23)
