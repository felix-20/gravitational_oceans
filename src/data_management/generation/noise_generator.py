import numpy as np

from src.data_management.generation.statistics import GONoise, GOStatisticsAll


class GONoiseGenerator:
    def __init__(self, statistics: GONoise = GOStatisticsAll().noise) -> None:
        self.constants = statistics

    def generate_static_noise(self):
        return self.constants.static.distribution(self.constants.static.mean, self.constants.static.std, (360, 5760))
