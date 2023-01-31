import numpy as np

from src.data_management.generation.statistics import GONoise, GOStatisticsAll


class GONoiseGenerator:
    def __init__(self, timesteps: list, statistics: GONoise = GOStatisticsAll().noise) -> None:
        self.constants = statistics
        self.timestep_count = len(timesteps)

    def generate_static_noise(self):
        return self.constants.static.distribution(self.constants.static.mean, self.constants.static.std, (360, self.timestep_count))
