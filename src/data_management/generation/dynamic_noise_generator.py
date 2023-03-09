# https://www.kaggle.com/code/vslaykovsky/g2net-realistic-simulation-of-test-noise

import numpy as np
from os.path import join
from secrets import token_hex
from cv2 import imwrite
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt


from src.data_management.generation.statistics import GODynamicNoise, GOStatistics
from src.helper.utils import PATH_TO_DYNAMIC_NOISE_FOLDER, PATH_TO_CACHE_FOLDER, print_red, print_blue, print_green, print_yellow
from src.data_management.generation.timestamps_generator import GOTimestepGenerator

class GODynamicNoiseGenerator:
    def __init__(self, timesteps: list, statistics: GOStatistics = GOStatistics(), bucket_count: int = 512) -> None:
        self.constants = statistics
        self.timestep_count = len(timesteps)
        self.timesteps = timesteps
        self.statistics = statistics
        self.dynamic_noise_stats = statistics.noise.dynamic
        self.bucket_count = bucket_count

    def linear_time_buckets(self):
        bucket_size = (self.timesteps[-1] - self.timesteps[0]) // self.bucket_count
        idx = np.searchsorted(self.timesteps, [self.timesteps[0] + bucket_size * i for i in range(self.bucket_count)])
        return list(idx) + [self.timestep_count]
    
    def generate_noise(self):
        bucket_delimiter_indices = self.linear_time_buckets()
        single_bucket_sizes = np.ediff1d(bucket_delimiter_indices)

        buckets = []

        for i in range(self.bucket_count):
            if single_bucket_sizes[i] == 0:
                continue

            bucket_std = self.dynamic_noise_stats.amplitude_std.sample()
            bucket_mean = self.dynamic_noise_stats.amplitude_mean.sample()

            bucket_mean_offset = self.dynamic_noise_stats.random_walk.sample() * np.random.choice([-1, 1], 1)

            buckets += [np.random.normal(bucket_mean_offset + bucket_mean, bucket_std, (360, single_bucket_sizes[i]))]

        return np.concatenate(buckets, axis=1)
    
    def __call__(self, idx: int):
        token = token_hex(3)
        amp = self.generate_noise()
        amp = (amp - amp.min())
        amp = amp * 255 / (amp.max())
        fname = join(PATH_TO_DYNAMIC_NOISE_FOLDER, f'{token}.png')
        imwrite(fname, amp)


def generate_sample(idx: int):
    np.random.seed(idx + int(time()))
    timesteps = GOTimestepGenerator().generate_timestamps()
    GODynamicNoiseGenerator(timesteps).__call__(idx)

if __name__ == '__main__':
    samples = 1000
    for i in tqdm(range(samples)):
        generate_sample(i)

    # with ThreadPoolExecutor() as t:
    #     for _ in tqdm(t.map(generate_sample, range(samples))):
    #         pass
    