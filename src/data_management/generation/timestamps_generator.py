from random import randint

import matplotlib.pyplot as plt

from src.data_management.generation.gap_generator import GOGapGenerator
from src.data_management.generation.statistics import GOStatistics, GOStatisticsTimestamp
from src.data_management.visualization import get_gaps


class GOTimestepGenerator:
    def __init__(self,
        statistics: GOStatistics = GOStatistics()) -> None:

        self.gap_statistics = statistics.gap
        self.constants = statistics.timestamps

    def generate_timestamps(self, start: int = None, gap_generator: GOGapGenerator = GOGapGenerator()):
        self.num_gaps = int(self.gap_statistics.count_distribution.sample())
        if not start:
            start = self.constants.start_distribution.sample() + self.constants.start_min
        gaps = gap_generator.generate_gaps(self.num_gaps)
        timestamps = [start]
        last_value = start
        for gap_size in gaps:
            timestamps += [last_value + gap_size]
            last_value = timestamps[-1]
        return timestamps


if __name__ == '__main__':
    time = GOTimestepGenerator().generate_timestamps()
    plt.hist(get_gaps(time, False), bins=200)
    plt.savefig('tmp/test_fig.png')
