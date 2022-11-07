import os
import numpy as np
import h5py

from src.helper.priority_queue import GOPriorityQueue
from src.helper.utils import PATH_TO_TEST_FOLDER, print_blue, print_green, print_red, print_yellow

PATH_TO_TMP_FOLDER = './tmp'
if not os.path.isdir(PATH_TO_TMP_FOLDER):
    os.makedirs(PATH_TO_TMP_FOLDER)

time_queue = GOPriorityQueue()
frequency_queue = GOPriorityQueue()
m = np.array([])

def process_file(path_to_file: str, label: bool) -> None:
    global m
    with h5py.File(path_to_file, 'r') as hd5_file:
        base_key = list(hd5_file.keys())[0]
        timestamps = hd5_file[f'{base_key}/H1/timestamps_GPS']
        frequencies = hd5_file[f'{base_key}/frequency_Hz']

        time_tuple = (timestamps[0], timestamps[-1])
        frequency_tuple = (frequencies[0], frequencies[-1])

        print('------------------')

        # TODO: are there overlapping timestamps or frequencies??
        x, is_duplicate_time = time_queue.insert(time_tuple)
        y, is_duplicate_frequency = frequency_queue.insert(frequency_tuple)
        print(m.shape)
        size = m.shape[-1] + 1

        if not is_duplicate_time:
            m = np.c_[m[:x], np.zeros(size),m[x:]]
        if not is_duplicate_frequency:
            tmp_m = np.transpose(m)
            size = tmp_m.shape[-1]
            tmp_m = np.c_[tmp_m[:x], np.zeros(size),tmp_m[x:]]
            m = np.transpose(tmp_m)
        
        m[x:y] = int(label) + 1
        print(x,y)
        print(m)
        input()


if not os.path.isfile(f'{PATH_TO_TEST_FOLDER}/train_labels.csv'):
    os.system(f'kaggle competitions download g2net-detecting-continuous-gravitational-waves -p {PATH_TO_TEST_FOLDER} -f train_labels.csv')
    assert os.path.isfile(f'{PATH_TO_TEST_FOLDER}/train_labels.csv'), 'Could not download train labels'


with open(f'{PATH_TO_TEST_FOLDER}/train_labels.csv', 'r') as file:
    next(file)
    i = 0
    for line in file:

        i += 1
        if i == 5:
            print_blue(time_queue)
            print_yellow(frequency_queue)
            exit(0)

        file_id, ground_truth = line.strip().split(',')
        if ground_truth == '-1':
            continue
        path_to_folder = PATH_TO_TEST_FOLDER
        path_to_folder += '/no_cw_hdf5' if ground_truth == '0' else '/cw_hdf5'
        final_file = f'{path_to_folder}/{file_id}.hdf5'

        if not os.path.isfile(final_file):
            os.system(f'kaggle competitions download g2net-detecting-continuous-gravitational-waves -p {PATH_TO_TMP_FOLDER} -f train/{file_id}.hdf5 --force --quiet')
            path_to_zip_file = f'{PATH_TO_TMP_FOLDER}/{file_id}.hdf5.zip'
            print_green(path_to_zip_file)
            os.system(f'unzip -qn -d {path_to_folder} {path_to_zip_file}')
            os.system(f'rm {path_to_zip_file}')

        process_file(final_file, ground_truth == '1')
