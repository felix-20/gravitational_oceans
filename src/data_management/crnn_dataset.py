from os import listdir, makedirs, path

import numpy as np
import torch
from tqdm import tqdm

from src.data_management.dataset import GODataset
from src.helper.utils import PATH_TO_TEST_FOLDER, print_green


class GOCRNNDataset(GODataset):

    def __init__(self, data_folder: str = PATH_TO_TEST_FOLDER, sequence_length: int = 5, save_folder: str = PATH_TO_TEST_FOLDER, transform=None):
        print_green('Crunching your data, hang on tight!')
        self.sequence_length = sequence_length
        self.save_folder = save_folder
        self.name_label_mapping = []
        self.path_to_mapping_file = path.join(self.save_folder, 'current_mapping.npy')

        if path.isdir(path.join(save_folder, 'no_cw_npy')) and path.isdir(path.join(save_folder, 'cw_npy')):
            # we have already converted the hdf5 files into numpy array files
            print('Loading from npy files')
            if path.isfile(self.path_to_mapping_file):
                self.name_label_mapping = np.load(self.path_to_mapping_file)
            else:
                self._create_new_mapping()
                self._save_mapping()
        else:
            print('Loading from hdf5 files')
            no_cw_folder = path.join(data_folder, 'no_cw_hdf5')
            cw_folder = path.join(data_folder, 'cw_hdf5')
            name_seed = 0
            name_seed = self._load_folder(no_cw_folder, 0, name_seed)
            name_seed = self._load_folder(cw_folder, 1, name_seed)
            self._leverage_data()
            self._save_mapping()

        np.random.shuffle(self.name_label_mapping)
        print_green('Dataset ready!')

    def _preprocess_stfts(self, stfts: np.array) -> np.array:
        time_samples = 360
        result = []

        total_frequencies, total_timesteps = stfts.shape

        time_frequency_amplitude_data = np.transpose(stfts, (1, 0))
        time_batch_count = total_timesteps // time_samples

        for time_batch_index in range(time_batch_count):
            time_frequency_batch = time_frequency_amplitude_data[time_batch_index * time_samples : (time_batch_index + 1) * time_samples]

            stft_image = []
            for frequency_amplitudes in time_frequency_batch:
                transformed = np.column_stack((frequency_amplitudes.real, frequency_amplitudes.imag, np.zeros(total_frequencies)))
                stft_image += [transformed]

            stft_image = np.transpose(np.array(stft_image), axes=(1, 0, 2))
            assert stft_image.shape == (360, 360, 3)
            result += [stft_image]

        return result

    def __len__(self) -> int:
        return len(self.name_label_mapping) // self.sequence_length

    def __getitem__(self, idx) -> dict:
        sequence_mapping = self.name_label_mapping[idx * self.sequence_length : (idx + 1) * self.sequence_length]
        sequence = self._load(sequence_mapping)

        data = []
        labels = []
        for stft_image, label in sequence:
            data += list(stft_image)
            labels += [label]

        return (np.transpose(np.array(data), (2, 1, 0)), torch.tensor(labels, dtype=torch.int32))

    def _create_new_mapping(self):
        print(f'Cannot find your mapping file in {self.path_to_mapping_file}')
        print(f'Creating new mapping file')
        all_no_cw_npy = [(int(name.strip('.npy')), 0) for name in listdir(path.join(self.save_folder, 'no_cw_npy'))]
        all_cw_npy = [(int(name.strip('.npy')), 1) for name in listdir(path.join(self.save_folder, 'cw_npy'))]
        self.name_label_mapping = np.array(all_cw_npy + all_no_cw_npy)
        self._leverage_data()
        np.random.shuffle(self.name_label_mapping)

    def _leverage_data(self) -> None:
        # make cw and no_cw data same size
        all_no_cw_data = [tup for tup in self.name_label_mapping if tup[1] == 0]
        all_cw_data = [tup for tup in self.name_label_mapping if tup[1] == 1]
        category_size = min(len(all_no_cw_data), len(all_cw_data))
        print(f'There are {category_size} files per label')
        self.name_label_mapping = np.array(all_no_cw_data[:category_size] + all_cw_data[:category_size])

    def _save(self, list_of_tuples: list, seed: int) -> list:
        no_cw_path = path.join(self.save_folder, 'no_cw_npy')
        cw_path = path.join(self.save_folder, 'cw_npy')
        makedirs(no_cw_path, exist_ok = True)
        makedirs(cw_path, exist_ok = True)

        list_of_names = []
        index = seed
        for stft_image, label in list_of_tuples:
            if label == 0:
                np.save(path.join(no_cw_path, str(index)), stft_image)
            elif label == 1:
                np.save(path.join(cw_path, str(index)), stft_image)
            list_of_names += [index]
            index += 1

        return list_of_names

    def _save_mapping(self) -> None:
        np.save(self.path_to_mapping_file, np.array(self.name_label_mapping))

    def _load(self, list_of_index: list) -> list:
        no_cw_path = path.join(self.save_folder, 'no_cw_npy')
        cw_path = path.join(self.save_folder, 'cw_npy')

        result = []
        for file_id, label in list_of_index:
            result += [(np.load(path.join(no_cw_path if label == 0 else cw_path, f'{file_id}.npy')), label)]

        return result

    def _load_folder(self, data_folder: str, label: int, name_seed: int) -> int:
        for file in tqdm(listdir(data_folder), f'Loading data for label {label}'):
            file_data = self._load_data_from_hdf5(path.join(data_folder, file))
            labeled_data = [(data, label) for data in file_data]
            self.name_label_mapping += [(name, label) for name in self._save(labeled_data, name_seed)]
            name_seed += len(labeled_data)
        return name_seed


if __name__ == '__main__':
    item = GOCRNNDataset().__getitem__(0)
    print(item[0].shape)
    print(item)
