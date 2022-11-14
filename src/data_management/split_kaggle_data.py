import os
from os import rename as move

from src.helper.utils import PATH_TO_TEST_FOLDER

path_to_raw_kaggle_data = os.path.join(PATH_TO_TEST_FOLDER, 'raw')

with open(f'{PATH_TO_TEST_FOLDER}/train_labels.csv', 'r') as file:
    next(file)
    i = 0
    for line in file:
        file_id, ground_truth = line.strip().split(',')
        if ground_truth == '-1':
            continue

        path_to_folder = PATH_TO_TEST_FOLDER
        path_to_folder += '/no_cw_hdf5' if ground_truth == '0' else '/cw_hdf5'
        dest_file_path = os.path.join(path_to_folder, f'{file_id}.hdf5')
        
        source_file_path = os.path.join(path_to_raw_kaggle_data, f'{file_id}.hdf5')
        move(source_file_path, dest_file_path)
