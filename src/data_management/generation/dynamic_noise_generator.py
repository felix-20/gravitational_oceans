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

    def generate_noise(self, bucket_count: int = 256, token: str='none'):
        bucket_size = self.timestep_count // bucket_count

        buckets = []

        # inicialize with a mean that is sampled from a distribution of bucket means
        #random_walk_value = self.dynamic_noise_stats.amplitude_mean.sample()
        #tmp = []
        for i in range(bucket_count):
            bucket_std = self.dynamic_noise_stats.amplitude_std.sample()
            bucket_mean = self.dynamic_noise_stats.random_walk.sample() * np.random.choice([-1, 1], 1)
            buckets += [np.random.normal(bucket_mean, bucket_std, (360, bucket_size))]

        # last bucket
        bucket_std = self.dynamic_noise_stats.amplitude_std.sample()
        bucket_mean = self.dynamic_noise_stats.random_walk.sample() * np.random.choice([-1, 1], 1)
        buckets += [np.random.normal(bucket_mean, bucket_std, (360, self.timestep_count - bucket_size * bucket_count))]


            #tmp += [self.dynamic_noise_stats.random_walk.sample() * np.random.choice([-1, 1], 1)]
            #print_yellow(buckets[-1])
            #exit(1)

        result = np.concatenate(buckets, axis=1)
        means = np.mean(result, axis=0)

        x = range(self.timestep_count)
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
    