from functools import cmp_to_key
from os import listdir, path
import csv

import numpy as np
from tqdm import tqdm

from src.helper.utils import PATH_TO_CACHE_FOLDER, PATH_TO_TEST_FOLDER, PATH_TO_TRAIN_FOLDER, PATH_TO_LABEL_FILE, open_hdf5_file, print_green


class GOHDF5Sorter:

    def __init__(self,
                 label_file: str = PATH_TO_LABEL_FILE,
                 train_folder: str = PATH_TO_TRAIN_FOLDER,
                 test_folder: str = PATH_TO_TEST_FOLDER):
        self.label_file = label_file
        self.train_folder = train_folder
        self.test_folder = test_folder

        self.sorted_by_frequency = []
        self.label_mapping = {}

        self.sorted_by_frequency_path = path.join(PATH_TO_CACHE_FOLDER, 'sorted_by_frequency.npy')

    def get_label_mapping(self):

        # if dict is filled -> return it
        if self.label_mapping != {}:
            return self.label_mapping

        with open(self.label_file, 'r') as csv_file:
            reader = csv.reader(csv_file)
            next(reader)
            for key, value in reader:
                train_file = path.join(self.train_folder, key + '.hdf5')

                if int(value) == -1:
                    value = 3

                self.label_mapping[train_file] = int(value) # 0, 1 or 3

        assert(len(self.label_mapping) == len(listdir(self.train_folder)))

        # set label EOS-Token for test-files
        for test_file in listdir(self.test_folder):
            self.label_mapping[test_file] = 3
        
        return self.label_mapping

    def get_sorted_frequencies(self):

        # if list is filled -> return it
        if len(self.sorted_by_frequency) != 0:
            return self.sorted_by_frequency

        # if file exists -> load it
        if path.isfile(self.sorted_by_frequency_path):
            self.sorted_by_frequency = np.load(self.sorted_by_frequency_path)
            return self.sorted_by_frequency


        self.get_label_mapping()

        print('Loading all data')
        frequency_lookup = self._preprocess_freq_ranges(self.label_mapping)

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
        self.sorted_by_frequency = self.label_mapping.items()
        self.sorted_by_frequency.sort(key=cmp_to_key(_compare_frequencies))

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
