import json
from os.path import isfile, dirname, abspath, join

class GOStatistics:
    def __init__(self, path_to_file: str = join(dirname(abspath(__file__)), 'statistics.json')) -> None:
        assert isfile(path_to_file), 'You need to have a statisics.json file'
        
        with open(path_to_file, 'r') as json_file:
            statistic_dict = json.load(json_file)
        
        assert 'gaps' in statistic_dict, 'You need to have a "gaps" keyword in your statistics file'
        self.gap = GOStatisticsGap(statistic_dict['gaps'])


class GOStatisticsGap:
    def __init__(self, gap_dict: dict) -> None:
        assert all(keyword in gap_dict for keyword in ['min', 'max', 'mean', 'median', 'std', '1800_ratio']), \
            'Your keywords in gap dict are missing some'        
        self.min = gap_dict['min']
        self.max = gap_dict['max']
        self.mean = gap_dict['mean']
        self.median = gap_dict['median']
        self.std = gap_dict['std']
        self.ratio = gap_dict['1800_ratio']


if __name__ == '__main__':
    print(GOStatistics().gap.max)
