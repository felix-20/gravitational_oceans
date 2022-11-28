from functools import cmp_to_key
from os import listdir, path

import numpy as np
from tqdm import tqdm

from src.helper.utils import PATH_TO_CACHE_FOLDER, PATH_TO_TEST_FOLDER, open_hdf5_file, print_green


class GOHDF5Sorter:

    def __init__(self, data_folder: str = PATH_TO_TEST_FOLDER):
        self.data_folder = data_folder

        self.sorted_by_frequency = []
        self.label_mapping = {}

        self.sorted_by_frequency_path = path.join(PATH_TO_CACHE_FOLDER, 'sorted_by_frequency.npy')
    
    def get_label_mapping(self):

        # if dict is filled -> return it
        if self.label_mapping != {}:
            return self.label_mapping

        no_cw_folder = path.join(self.data_folder, 'no_cw_hdf5')
        cw_folder = path.join(self.data_folder, 'cw_hdf5')

        file_label_mapping = [(path.join(no_cw_folder, file_name), 0) for file_name in listdir(no_cw_folder)]
        file_label_mapping += [(path.join(cw_folder, file_name), 1) for file_name in listdir(cw_folder)]

        self.label_mapping = dict(file_label_mapping)
        return self.label_mapping

    def get_sorted_frequencies(self):

        # if list is filled -> return it
        if len(self.sorted_by_frequency) != 0:
            return self.sorted_by_frequency

        # if file exists -> load it
        if path.isfile(self.sorted_by_frequency_path):
            self.sorted_by_frequency = np.load(self.sorted_by_frequency_path)
            return self.sorted_by_frequency


        no_cw_folder = path.join(self.data_folder, 'no_cw_hdf5')
        cw_folder = path.join(self.data_folder, 'cw_hdf5')

        file_label_mapping = [(path.join(no_cw_folder, file_name), 0) for file_name in listdir(no_cw_folder)]
        file_label_mapping += [(path.join(cw_folder, file_name), 1) for file_name in listdir(cw_folder)]

        print('Loading all data')
        frequency_lookup = self._preprocess_freq_ranges(file_label_mapping)

        def _compare_frequencies(file_a, file_b):
            data_a = frequency_lookup[file_a[0]]
            data_b = frequency_lookup[file_b[0]]

            if data_a < data_b:
                return -1
            elif data_a > data_b:
                return 1
            else:
                return 0

        print('Sorting data by frequency')
        file_label_mapping.sort(key=cmp_to_key(_compare_frequencies))
        self.sorted_by_frequency = file_label_mapping

        np.save(self.sorted_by_frequency_path, self.sorted_by_frequency)

        return self.sorted_by_frequency

    def _preprocess_freq_ranges(self, file_label_mapping):
        result = {}
        for path, _ in tqdm(file_label_mapping):
            result[path] = open_hdf5_file(path)['frequencies'][0]
        return result


if __name__ == '__main__':
    sorter = GOHDF5Sorter()
    sorted_res = sorter.get_sorted_frequencies()
    # print_green(sorted_res)
