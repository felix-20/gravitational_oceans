from concurrent.futures import ProcessPoolExecutor
from os.path import join
from secrets import token_hex

import numpy as np
from cv2 import imwrite
from tqdm import tqdm

from src.data_management.generation.statistics import GOStaticNoise, GOStatistics
from src.data_management.generation.timestamps_generator import GOTimestepGenerator
from src.helper.utils import PATH_TO_STATIC_NOISE_FOLDER


class GOStaticNoiseGenerator:
    def __init__(self, timesteps: list, statistics: GOStaticNoise = GOStatistics().noise.static) -> None:
        self.constants = statistics
        self.timestep_count = len(timesteps)

    def __call__(self, idx):
        token = token_hex(3)
        amp = self.constants.distribution.sample((360, self.timestep_count))
        amp = (amp - amp.min())
        amp = amp * 255 / (amp.max())
        fname = join(PATH_TO_STATIC_NOISE_FOLDER, f'{token}.png')
        imwrite(fname, amp)

    def generate_noises(self, n: int):
        with ProcessPoolExecutor() as p:
            for _ in tqdm(p.map(self, range(n))):
                pass

if __name__ == '__main__':
    timesteps = GOTimestepGenerator().generate_timestamps()
    samples = 100
    GOStaticNoiseGenerator(timesteps).generate_noises(samples)
