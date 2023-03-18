import json
from os.path import abspath, dirname, isfile, join

import numpy as np

from src.data_management.generation.distribution import GODistributionFactory
from src.helper.utils import print_yellow


class GOStatistics:
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

        if 'frequencies' in statistic_dict:
            self.frequencies = GOFrequencies(statistic_dict['frequencies'])


class GOStatisticsGap:
    def __init__(self, gap_dict: dict) -> None:
        self.count_distribution = GODistributionFactory.parse(gap_dict['count'])
        # self.count_mean = gap_dict['count_mean']
        # self.count_std = gap_dict['count_std']

        self.distribution = GODistributionFactory.parse(gap_dict['gap'])
        self.min = gap_dict['min']
        self.max = gap_dict['max']
        # self.mean = gap_dict['mean']
        # self.std = gap_dict['std']
        self.median = gap_dict['median']
        self.ratio = gap_dict['1800_ratio']


class GOStatisticsTimestamp:
    def __init__(self, timestamps_dict: dict) -> None:
        assert all(keyword in timestamps_dict for keyword in ['start_mean', 'start_distribution', 'start_min']), \
            'Your keywords in timestamps dict are missing some'

        start_dict = {}
        start_dict['distribution'] = timestamps_dict['start_distribution']
        start_dict['mean'] = timestamps_dict['start_mean']

        self.start_distribution = GODistributionFactory.parse(start_dict)
        self.start_min = timestamps_dict['start_min']


class GONoise:
    def __init__(self, noise_dict: dict) -> None:
        assert all(keyword in noise_dict for keyword in ['static', 'dynamic']), \
            'Your keywords in noise dict are missing some'
        self.static = GOStaticNoise(noise_dict['static'])
        self.dynamic = GODynamicNoise(noise_dict['dynamic'])


class GODynamicNoise:
    def __init__(self, dynamic_dict: dict) -> None:
        self.amplitude_mean = GODistributionFactory.parse(dynamic_dict['amplitude_mean'])
        self.amplitude_std = GODistributionFactory.parse(dynamic_dict['amplitude_std'])
        self.random_walk = GODistributionFactory.parse(dynamic_dict['random_walk'])


class GOStaticNoise:
    def __init__(self, noise_dict) -> None:
        assert all(keyword in noise_dict for keyword in ['mean', 'std', 'distribution']), \
            'Your keywords in dynamic noise dict are missing some'

        self.mean = GODistributionFactory.parse(noise_dict['mean'])
        self.std = GODistributionFactory.parse(noise_dict['std'])

        noise_dict['mean'] = self.mean
        noise_dict['std'] = self.std
        self.distribution = GODistributionFactory.parse(noise_dict)

class GOFrequencies:
    def __init__(self, frequency_dict: dict) -> None:
        self.distribution = GODistributionFactory.parse(frequency_dict)


if __name__ == '__main__':
    print(GOStatistics().noise.static.distribution)
