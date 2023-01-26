import json
from os.path import abspath, dirname, isfile, join

import numpy as np

from src.helper.utils import print_yellow


class GOStatistics:
    def translate_distributions(self, distribution_keyword: str):
        if distribution_keyword == 'exponential':
            return np.random.exponential
        elif distribution_keyword == 'normal':
            return np.random.normal
        else:
            assert False, f'Your given distribution {distribution_keyword} is not valid'


class GOStatisticsAll:
    def __init__(self, path_to_file: str = join(dirname(abspath(__file__)), 'statistics.json')) -> None:
        assert isfile(path_to_file), 'You need to have a statisics.json file'

        with open(path_to_file, 'r') as json_file:
            statistic_dict = json.load(json_file)

        if 'gaps' in statistic_dict:
            self.gap = GOStatisticsGap(statistic_dict['gaps'])
        else:
            print_yellow('You did not provide the "gaps" keyword')

        if 'timestamps' in statistic_dict:
            self.timestamps = GOStatisticsTimestamp(statistic_dict['timestamps'])
        else:
            print_yellow('You did not provide the "timestamps" keyword')
        
        if 'noise' in statistic_dict:
            self.noise = GONoise(statistic_dict['noise'])


class GOStatisticsGap(GOStatistics):
    def __init__(self, gap_dict: dict) -> None:
        assert all(keyword in gap_dict for keyword in \
            ['min', 'max', 'mean', 'median', 'std', '1800_ratio', 'distribution', 'count_mean', 'count_std', 'count_distribution']), \
            'Your keywords in gap dict are missing some'
        self.count_mean = gap_dict['count_mean']
        self.count_std = gap_dict['count_std']
        self.count_distribution = self.translate_distributions(gap_dict['count_distribution'])

        self.distribution = self.translate_distributions(gap_dict['distribution'])
        self.min = gap_dict['min']
        self.max = gap_dict['max']
        self.mean = gap_dict['mean']
        self.median = gap_dict['median']
        self.std = gap_dict['std']
        self.ratio = gap_dict['1800_ratio']


class GOStatisticsTimestamp(GOStatistics):
    def __init__(self, timestamps_dict: dict) -> None:
        assert all(keyword in timestamps_dict for keyword in ['start_mean', 'start_distribution', 'start_min']), \
            'Your keywords in timestamps dict are missing some'

        self.start_mean = timestamps_dict['start_mean']
        self.start_distribution = self.translate_distributions(timestamps_dict['start_distribution'])
        self.start_min = timestamps_dict['start_min']


class GONoise(GOStatistics):
    def __init__(self, noise_dict: dict) -> None:
        assert all(keyword in noise_dict for keyword in ['static', 'dynamic']), \
            'Your keywords in noise dict are missing some'
        self.static = GOStaticNoise(noise_dict['static'])
        self.dynamic = GODynamicNoise(noise_dict['dynamic'])


class GODynamicNoise(GOStatistics):
    def __init__(self, dynamic_dict: dict) -> None:
        pass


class GOStaticNoise(GOStatistics):
    def __init__(self, noise_dict) -> None:
        assert all(keyword in noise_dict for keyword in ['static', 'dynamic']), \
            'Your keywords in dynamic noise dict are missing some'

        self.mean = noise_dict['mean']
        self.std = noise_dict['std']
        self.distribution = self.translate_distributions(noise_dict['distribution'])


if __name__ == '__main__':
    print(GOStatistics().gap.max)
