import numpy as np

from src.data_management.generation.statistics import GONoise, GOStatistics


class GONoiseGenerator:
    def __init__(self, timesteps: list, statistics: GONoise = GOStatistics().noise) -> None:
        self.constants = statistics
        self.timestep_count = len(timesteps)

    def generate_static_noise(self):
        return self.constants.static.distribution.distribute((360, self.timestep_count))
