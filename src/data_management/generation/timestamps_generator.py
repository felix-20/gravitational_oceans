import matplotlib.pyplot as plt
from random import randint

from src.data_management.generation.gap_generator import GOGapGenerator
from src.data_management.visualization import get_gaps


class GOTimestepGenerator:
    def __init__(self, number_of_gaps: int = None) -> None:
        self.num_gaps = number_of_gaps
        if not number_of_gaps:
            self.num_gaps = randint(4000, 5000)

    def generate_timestamps(self, start: int = None, gap_generator: GOGapGenerator = GOGapGenerator()):
        if not start:
            start = randint(4000, 5000)
        gaps = gap_generator.generate_gaps(self.num_gaps)
        timestamps = [start]
        last_value = start
        for gap_size in gaps:
            timestamps += [last_value + gap_size]
            last_value = timestamps[-1]
        return timestamps


if __name__ == '__main__':
    time = GOTimestepGenerator(7000).generate_timestamps()
    plt.hist(get_gaps(time, False), bins=200)
    plt.savefig('tmp/test_fig.png')
