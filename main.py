import os

from kaggle.api.kaggle_api_extended import KaggleApi

from data_generator import GODataGenerator
from utils import PATH_TO_TEST_FOLDER, PATH_TO_TRAIN_FOLDER, print_green




api = KaggleApi()
api.authenticate()

# download single file
# Signature: dataset_download_file(dataset, file_name, path=None, force=False, quiet=True)
#
# api.dataset_download_file('g2net-detecting-continuous-gravitational-waves', 'train/01dba9731.hdf5', path=None, force=False, quiet=True)
# all_files = api.competitions_data_list_files('g2net-detecting-continuous-gravitational-waves')
# with open('./data/all_files.txt', 'w') as file:
#     for f in all_files:
#         file.write(str(f) + '\n')

os.system(f'kaggle competitions download g2net-detecting-continuous-gravitational-waves -p {PATH_TO_TEST_FOLDER} -f train_labels.csv')
assert os.path.isfile(f'{PATH_TO_TEST_FOLDER}/train_labels.csv'), 'Could not download train labels'


with open(f'{PATH_TO_TEST_FOLDER}/train_labels.csv', 'r') as file:
    next(file)
    for line in file:
        file_id, ground_truth = line.strip().split(',')
        if ground_truth == '-1':
            continue
        path_to_folder = PATH_TO_TEST_FOLDER
        path_to_folder += '/no_cw_hdf5' if ground_truth == '0' else '/cw_hdf5'
        os.system(f'kaggle competitions download g2net-detecting-continuous-gravitational-waves -p {path_to_folder} -f train/{file_id}.hdf5')
        path_to_file = f'{path_to_folder}/{file_id}.hdf5.zip'
        print_green(path_to_file)
        os.system(f'unzip -d {path_to_folder} {path_to_file}')
        os.system(f'rm {path_to_file}')
