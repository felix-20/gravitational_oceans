from random import randint

import matplotlib.pyplot as plt

from src.data_management.generation.gap_generator import GOGapGenerator
from src.data_management.generation.statistics import GOStatistics, GOStatisticsTimestamp
from src.data_management.visualization import get_gaps


class GOTimestepGenerator:
    def __init__(self, number_of_gaps: int = None,
        statistics: GOStatistics = GOStatistics()) -> None:

        self.num_gaps = number_of_gaps
        if not number_of_gaps:
            self.num_gaps = int(statistics.gap.count_distribution.distribute())
        self.constants = statistics.timestamps

    def generate_timestamps(self, start: int = None, gap_generator: GOGapGenerator = GOGapGenerator()):
        if not start:
            start = self.constants.start_distribution.distribute() + self.constants.start_min
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
