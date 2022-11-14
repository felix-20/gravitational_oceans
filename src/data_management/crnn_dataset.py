import torch
import numpy as np
from random import shuffle
from os import path, listdir, makedirs
from tqdm import tqdm

from src.helper.utils import PATH_TO_TEST_FOLDER, print_green
from src.data_management.dataset import GODataset

class GOCRNNDataset(GODataset):

    def __init__(self, data_folder: str = PATH_TO_TEST_FOLDER, sequence_length: int = 5, save_folder: str = PATH_TO_TEST_FOLDER, transform=None):
        print_green('Crunching your data, hang on tight!')
        self.sequence_length = sequence_length
        self.save_folder = save_folder
        self.name_label_mapping = []

        if path.isdir(path.join(save_folder, 'no_cw_npy')) and path.isdir(path.join(save_folder, 'cw_npy')):
            print('Loading from npy files')
            # TODO: change load to load list of tuples aka new matchinh
            self._load()
        else:
            print('Loading from hdf5 files')
            name_seed = 0
            no_cw_folder = path.join(data_folder, 'no_cw_hdf5')
            cw_folder = path.join(data_folder, 'cw_hdf5')
            name_seed = self._load_folder(no_cw_folder, 0, name_seed)
            name_seed = self._load_folder(cw_folder, 1, name_seed)

            # TODO: make data same size
            no_cw_data = []
            cw_data = []
            for element in self.frame:
                match element[1]:
                    case 0:
                        no_cw_data += [element]
                    case 1:
                        cw_data += [element]

            category_size = min(len(no_cw_data), len(cw_data))
            no_cw_data = no_cw_data[:category_size]
            cw_data = cw_data[:category_size]

            self.frame = no_cw_data + cw_data
            self._save()
        
        shuffle(self.frame)

        print_green('Dataset ready!')

        #self.frame = [item for item in self.frame for _ in range(30)]

    def _load_folder(self, data_folder: str, label: int, name_seed: int) -> int:
        for file in listdir(data_folder):
            file_data = self._load_data_from_hdf5(path.join(data_folder, file))
            labeled_data = [(data, label) for data in file_data]
            self.name_label_mapping += [(name, label) for name in self._save(labeled_data, name_seed)]
            name_seed += len(labeled_data)
        return name_seed

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
        return len(self.frame) // self.sequence_length

    def __getitem__(self, idx) -> dict:
        sequence = self.frame[idx * self.sequence_length : (idx + 1) * self.sequence_length]
        data = []
        labels = []

        for stft_image, label in sequence:
            data += list(stft_image)
            labels += [label]

        return (np.transpose(np.array(data), (2, 1, 0)), torch.tensor(labels, dtype=torch.int32))
        # return (np.transpose(self.frame[idx][0], (0, 2, 1)), torch.tensor([self.frame[idx][1]], dtype=torch.int32))

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
    
    def _load(self) -> None:
        no_cw_path = path.join(self.save_folder, 'no_cw_npy')
        cw_path = path.join(self.save_folder, 'cw_npy')

        self.frame = []

        for npy_file in tqdm(listdir(no_cw_path), 'loading no cw files'):
            self.frame += [(np.load(path.join(no_cw_path, npy_file)), 0)]

        for npy_file in tqdm(listdir(cw_path), 'loading cw files'):
            self.frame += [(np.load(path.join(cw_path, npy_file)), 1)]
        

if __name__ == '__main__':
    item = GOCRNNDataset().__getitem__(0)
    print(item[0].shape)
    print(item)
