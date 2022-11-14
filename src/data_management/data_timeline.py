import os

import h5py
import numpy as np

from src.helper.priority_queue import GOPriorityQueue
from src.helper.utils import PATH_TO_TEST_FOLDER, print_blue, print_green, print_red, print_yellow

PATH_TO_TMP_FOLDER = './tmp'
if not os.path.isdir(PATH_TO_TMP_FOLDER):
    os.makedirs(PATH_TO_TMP_FOLDER)

time_queue = GOPriorityQueue()
frequency_queue = GOPriorityQueue()
matrix = np.array([])


def append_row(m, i):
    if m.size == 0:
        return np.zeros((1))

    size = m.shape[0]

    #print_red(m)
    result = None
    if i == 0:
        result = np.r_[[np.zeros(1)], m]
    elif i == size:
        result = np.r_[m, [np.zeros(1)]]
    else:
        result = np.r_[m[:i], [np.zeros(1)], m[i:]]
    #print_red(result)
    return result


def process_file(path_to_file: str, label: bool, is_first: bool) -> None:
    global matrix
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
        print_red(f't_{is_duplicate_time}, f_{is_duplicate_frequency}')
        print_green(f'{x}, {y}')

        if is_first:
            matrix = np.empty((1,1))
            matrix.fill(int(ground_truth == '1') + 1)
            return

        if not is_duplicate_time:
            matrix = append_row(matrix, x)
            print_blue(matrix)
        if not is_duplicate_frequency:
            tmp_m = np.transpose(matrix)
            tmp_m = append_row(tmp_m, y)
            matrix = np.transpose(tmp_m)
            print_blue(matrix)

        matrix[x,y] = int(label) + 1
        print(x,y)
        print(matrix)
        #input()


if not os.path.isfile(f'{PATH_TO_TEST_FOLDER}/train_labels.csv'):
    os.system(f'kaggle competitions download g2net-detecting-continuous-gravitational-waves -p {PATH_TO_TEST_FOLDER} -f train_labels.csv')
    assert os.path.isfile(f'{PATH_TO_TEST_FOLDER}/train_labels.csv'), 'Could not download train labels'


with open(f'{PATH_TO_TEST_FOLDER}/train_labels.csv', 'r') as file:
    next(file)
    i = 0
    for line in file:

        i += 1
        if i == 150:
            print_blue(time_queue)
            print_yellow(frequency_queue)
            print_red(len(frequency_queue.queue))
            exit(0)

        file_id, ground_truth = line.strip().split(',')
        if ground_truth == '-1':
            continue
        path_to_folder = PATH_TO_TEST_FOLDER
        path_to_folder += '/no_cw_hdf5' if ground_truth == '0' else '/cw_hdf5'
        final_file = f'{path_to_folder}/{file_id}.hdf5'

        if not os.path.isfile(final_file):
            print_blue(f'downloading: {file_id}.hdf5')
            os.system(f'kaggle competitions download g2net-detecting-continuous-gravitational-waves -p {PATH_TO_TMP_FOLDER} -f train/{file_id}.hdf5 --force --quiet')
            path_to_zip_file = f'{PATH_TO_TMP_FOLDER}/{file_id}.hdf5.zip'
            print_green(path_to_zip_file)
            os.system(f'unzip -qn -d {path_to_folder} {path_to_zip_file}')
            os.system(f'rm {path_to_zip_file}')

        process_file(final_file, ground_truth == '1', i == 1)
