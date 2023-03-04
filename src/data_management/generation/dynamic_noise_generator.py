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
    def __init__(self, timesteps: list, statistics: GOStatistics = GOStatistics(), buckets: int = 256) -> None:
        self.constants = statistics
        self.timestep_count = len(timesteps)
        self.timesteps = timesteps
        self.statistics = statistics
        self.dynamic_noise_stats = statistics.noise.dynamic
        self.buckets = buckets

    def linear_time_buckets(self, bucket_count):
        bucket_size = (self.timesteps[-1] - self.timesteps[0]) // bucket_count
        idx = np.searchsorted(self.timesteps, [self.timesteps[0] + bucket_size * i for i in range(bucket_count)])
        return list(idx) + [self.timestep_count]
    
    def bucketize_real_noise_asd(self, sfts, bucket_count=256):
        bucket_size = (self.timesteps[-1] - self.timesteps[0]) // bucket_count
        idx = np.searchsorted(self.timesteps, [self.timesteps[0] + bucket_size * i for i in range(bucket_count)])
        global_noise_amp = np.mean(np.abs(sfts))
        return np.array([np.mean(np.abs(i)) if i.shape[1] > 0 else global_noise_amp for i in np.array_split(sfts, idx[1:], axis=1)]), bucket_size

    def generate_noise(self, bucket_count: int = 512, token: str='none'):
        bucket_delimiter_indices = self.linear_time_buckets(bucket_count)
        single_bucket_sizes = np.ediff1d(bucket_delimiter_indices)

        buckets = []

        for i in range(bucket_count):
            if single_bucket_sizes[i] == 0:
                continue

            bucket_std = self.dynamic_noise_stats.amplitude_std.sample()
            bucket_mean = self.dynamic_noise_stats.amplitude_mean.sample()

            bucket_mean_offset = self.dynamic_noise_stats.random_walk.sample() * np.random.choice([-1, 1], 1)

            buckets += [np.random.normal(bucket_mean_offset + bucket_mean, bucket_std, (360, single_bucket_sizes[i]))]

        # generate statistics
        result = np.concatenate(buckets, axis=1)
        #means = [np.mean(bucket) for bucket in np.array_split(result, bucket_count, axis=1)]
        #means = np.mean(result, axis=0)
        means, _ = self.bucketize_real_noise_asd(result, bucket_count)
        x = range(bucket_count)#range(self.timestep_count)
        plt.plot(x, means)
        plt.savefig(join(PATH_TO_CACHE_FOLDER, 'direct_statistics_'+token+'.png'))
        plt.clf()

        return np.concatenate(buckets, axis=1)
    
    def __call__(self, idx: int):

        token = token_hex(3)
        amp = self.generate_noise(token=token)
        amp = (amp - amp.min())
        amp = amp * 255 / (amp.max())
        fname = join(PATH_TO_DYNAMIC_NOISE_FOLDER, f'{token}.png')
        imwrite(fname, amp)


def generate_sample(idx: int):
    np.random.seed(idx + int(time()))
    timesteps = GOTimestepGenerator().generate_timestamps()
    GODynamicNoiseGenerator(timesteps).__call__(idx)

if __name__ == '__main__':
    samples = 10
    for i in tqdm(range(samples)):
        generate_sample(i)

    # with ThreadPoolExecutor() as t:
    #     for _ in tqdm(t.map(generate_sample, range(samples))):
    #         pass
    