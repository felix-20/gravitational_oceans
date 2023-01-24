import matplotlib.pyplot as plt
import numpy as np

from src.data_management.generation.statistics import GOStatistics
from src.data_management.visualization import get_gaps


class GOGapGenerator:
    def __init__(self, statistics: GOStatistics = GOStatistics(), start_timestamp: int = 1) -> None:
        self.constants = statistics
        self.start = start_timestamp

    def generate_gaps(self, n: int) -> np.array:
        gaps = []
        for _ in range(n):
            if np.random.uniform() < self.constants.gap.ratio:
                # 1800 case
                gaps += [1800]
            else:
                # exponential distribution in all other cases
                gaps += [np.random.exponential(self.constants.gap.mean)]
        timestamps = [self.start]
        last_value = self.start
        for gap_size in gaps:
            timestamps += [last_value + gap_size]
            last_value = timestamps[-1]
        return timestamps


if __name__ == '__main__':
    time = GOGapGenerator().generate_gaps(1000000)
    plt.hist(get_gaps(time, False))
    plt.savefig('tmp/test_fig.png')
