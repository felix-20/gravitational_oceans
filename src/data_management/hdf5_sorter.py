from os import path, listdir
import numpy as np
from functools import cmp_to_key

from src.helper.utils import PATH_TO_TEST_FOLDER, PATH_TO_CACHE_FOLDER, print_green, open_hdf5_file

class GOHDF5Sorter:

    def __init__(self, data_folder: str = PATH_TO_TEST_FOLDER):
        self.data_folder = data_folder

        self.sorted_by_frequency = []
        self.label_mapping = []

        self.sorted_by_frequency_path = path.join(PATH_TO_CACHE_FOLDER, 'sorted_by_frequency.npy')

    def get_sorted_frequencies(self):

        # if list is filled -> return it
        if self.sorted_by_frequency:
            return self.sorted_by_frequency

        # if file exists -> load it
        if path.isfile(self.sorted_by_frequency_path):
            self.sorted_by_frequency = np.load(self.sorted_by_frequency_path)
            return self.sorted_by_frequency

        
        no_cw_folder = path.join(self.data_folder, 'no_cw_hdf5')
        cw_folder = path.join(self.data_folder, 'cw_hdf5')

        file_label_mapping = []
        file_label_mapping += [(path.join(no_cw_folder, file_name), 0) for file_name in listdir(no_cw_folder)]
        file_label_mapping += [(path.join(cw_folder, file_name), 1) for file_name in listdir(cw_folder)]

        def _compare_frequencies(file_a, file_b):
            data_a = open_hdf5_file(file_a[0])
            data_b = open_hdf5_file(file_b[0])

            if data_a['frequencies'][0] < data_b['frequencies'][0]:
                return -1
            elif data_a['frequencies'][0] > data_b['frequencies'][0]:
                return 1
            else:
                return 0

        file_label_mapping.sort(key=cmp_to_key(_compare_frequencies))
        self.sorted_by_frequency = file_label_mapping

        np.save(self.sorted_by_frequency_path, self.sorted_by_frequency)

        return self.sorted_by_frequency


if __name__ == '__main__':
    sorter = GOHDF5Sorter()
    print_green(sorter.get_sorted_frequencies())
