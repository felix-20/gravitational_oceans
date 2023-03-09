import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from os.path import join
from secrets import token_hex
from cv2 import imwrite
from time import time

from src.data_management.generation.statistics import GOStaticNoise, GOStatistics
from src.helper.utils import PATH_TO_STATIC_NOISE_FOLDER
from src.data_management.generation.timestamps_generator import GOTimestepGenerator

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

def generate_sample(idx: int):
    np.random.seed(idx + int(time()))
    timesteps = GOTimestepGenerator().generate_timestamps()
    GOStaticNoiseGenerator(timesteps).__call__(idx)

if __name__ == '__main__':
    samples = 1000
    for i in tqdm(range(samples)):
        generate_sample(i)