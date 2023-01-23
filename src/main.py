import os

from kaggle.api.kaggle_api_extended import KaggleApi

from helper.utils import PATH_TO_TEST_FOLDER, PATH_TO_TRAIN_FOLDER, print_green

# download single file
# Signature: dataset_download_file(dataset, file_name, path=None, force=False, quiet=True)
#
# api.dataset_download_file('g2net-detecting-continuous-gravitational-waves', 'train/01dba9731.hdf5', path=None, force=False, quiet=True)
# all_files = api.competitions_data_list_files('g2net-detecting-continuous-gravitational-waves')
# with open('./data/all_files.txt', 'w') as file:
#     for f in all_files:
#         file.write(str(f) + '\n')

#os.system(f'kaggle competitions download g2net-detecting-continuous-gravitational-waves -p {PATH_TO_TEST_FOLDER} -f train_labels.csv')
#assert os.path.isfile(f'{PATH_TO_TEST_FOLDER}/train_labels.csv'), 'Could not download train labels'


with open('test_files.txt', 'r') as file:
    for file_name in file:
        os.system(f'kaggle competitions download g2net-detecting-continuous-gravitational-waves -p tmp -f test/{file_name}.hdf5')
        path_to_file = f'tmp/{file_name}.zip'
        os.system(f'unzip -d {PATH_TO_TEST_FOLDER} {path_to_file}')
        os.system(f'rm {path_to_file}')
