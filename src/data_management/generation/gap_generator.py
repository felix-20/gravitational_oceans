import matplotlib.pyplot as plt
import numpy as np

from src.data_management.generation.statistics import GOStatisticsAll, GOStatisticsGap
from src.data_management.visualization import get_gaps


class GOGapGenerator:
    def __init__(self, statistics: GOStatisticsGap = GOStatisticsAll().gap) -> None:
        self.constants = statistics

    def generate_gaps(self, n: int) -> np.array:
        gaps = []
        for _ in range(n):
            if np.random.uniform() < self.constants.ratio:
                # 1800 case
                gaps += [1800]
            else:
                # exponential distribution in all other cases
                gaps += [self.constants.distribution(self.constants.mean)]
        return gaps
