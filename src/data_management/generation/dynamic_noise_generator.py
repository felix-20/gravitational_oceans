# https://www.kaggle.com/code/vslaykovsky/g2net-realistic-simulation-of-test-noise

import pyfstat
from pyfstat.utils import get_sft_as_arrays
import copy
import numpy as np
from multiprocess import Pool
from os import makedirs
from os.path import join
from shutil import rmtree
from secrets import token_hex
from cv2 import imwrite
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from time import time

from src.data_management.generation.statistics import GODynamicNoise, GOStatistics
from src.helper.utils import PATH_TO_DYNAMIC_NOISE_FOLDER, PATH_TO_CACHE_FOLDER
from src.data_management.generation.timestamps_generator import GOTimestepGenerator
from src.helper.utils import print_red, print_blue, print_green, print_yellow

class GODynamicNoiseGenerator:
    def __init__(self, timesteps: list, statistics: GOStatistics = GOStatistics(), buckets: int = 256) -> None:
        self.constants = statistics
        self.timestep_count = len(timesteps)
        self.timesteps = timesteps
        self.statistics = statistics
        self.dynamic_noise_stats = statistics.noise.dynamic

        self.buckets = buckets
        self.c_sqrsx = 26.5

    def bucketize_real_noise_asd(self, buckets=256):
        bucket_size = (self.timesteps[-1] - self.timesteps[0]) // buckets
        #idx = np.searchsorted(self.timesteps, [self.timesteps[0] + bucket_size * i for i in range(buckets)])
        #global_noise_amplitude = self.dynamic_noise_stats.amplitude_distribution.sample()
        #np.array([np.mean(np.abs(i)) if i.shape[1] > 0 else global_noise_amplitude for i in np.array_split(sfts, idx[1:], axis=1)])
        #return np.fromfunction(self.dynamic_noise_stats.amplitude_distribution.sample, self.timestep_count),  bucket_size
        return np.array(list(map(self.dynamic_noise_stats.amplitude_distribution.sample, np.full(self.timestep_count, None)))), bucket_size

    def generate_segment(self, writer_kwargs):
        writer = pyfstat.Writer(**writer_kwargs)
        writer.make_data()
        return writer.sftfilepath


    def simulate_real_noise(self, tmp_folder):
        asd, bucket_size = self.bucketize_real_noise_asd(buckets=self.buckets)
        print_green(asd)
        writer_kwargs = {
            "outdir": tmp_folder,
            "detectors": 'L1,H1',  # Detector to simulate, in this case LIGO Hanford
            "F0": self.statistics.frequencies.distribution.sample(),  # Central frequency of the band to be generated [Hz]
            'Band': 1/5.01,  # Frequency band-width around F0 [Hz]
            "Tsft": 1800,  # Fourier transform time duration
            "SFTWindowType": "tukey",
            "SFTWindowBeta": 0.01,
            "duration": bucket_size,
            'timestamps': {
                'H1': np.array(self.timesteps, dtype=int),
                'L1': np.array(self.timesteps, dtype=int),
            }
        }

        all_args = []
        for segment in range(self.buckets):
            args = copy.deepcopy(writer_kwargs)
            args["label"] = f"segment_{segment}"
            args["sqrtSX"] = asd[segment] / self.c_sqrsx
            args["tstart"] = self.timesteps[0] + segment * bucket_size
            all_args.append(args)

        with Pool() as p:
            sft_path = p.map(self.generate_segment, all_args)

        sft_path = ";".join(sorted(sft_path))  # Concatenate different files using ;
        _, _, fourier_data = get_sft_as_arrays(sft_path)
        
        print_red(fourier_data.shape)
        return fourier_data[:,-360:, :]
    
    def __call__(self, idx: int):
        path_to_tmp_folder = join(PATH_TO_CACHE_FOLDER, 'signal_generation', str(idx))
        makedirs(path_to_tmp_folder, exist_ok=True)

        amplitudes = self.simulate_real_noise(path_to_tmp_folder)
        rmtree(path_to_tmp_folder)

        token = token_hex(3)
        for det in amplitudes:
            amp = np.abs(amplitudes[det][-360:, :])
            amp = (amp - amp.min())
            amp = amp * 255 / (amp.max())
            fname = join(PATH_TO_DYNAMIC_NOISE_FOLDER, f'{det}_{token}.png')
            imwrite(fname, amp)

def generate_sample(idx: int):
    np.random.seed(idx + int(time()))
    timesteps = GOTimestepGenerator().generate_timestamps()
    GODynamicNoiseGenerator(timesteps).__call__(idx)

if __name__ == '__main__':
    samples = 1
    with ThreadPoolExecutor() as t:
        for _ in tqdm(t.map(generate_sample, range(samples))):
            pass
    