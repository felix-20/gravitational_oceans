import numpy as np


class GODistribution:
    def __init__(self) -> None:
        pass

    def distribute(self, size=None):
        pass


class GONormalDistribution(GODistribution):
    def __init__(self, parameter_dict: dict) -> None:
        self.mean = parameter_dict['mean']
        self.std = parameter_dict['std']

    def distribute(self, size=None):
        return np.random.normal(self.mean, self.std, size)


class GOConstDistribution(GODistribution):
    def __init__(self, parameter_dict: dict) -> None:
        self.const = parameter_dict['const']

    def distribute(self, size=None):
        return np.full(size, self.const) if size else self.const


class GOExponentialDistribution(GODistribution):
    def __init__(self, parameter_dict: dict) -> None:
        self.mean = parameter_dict['mean']

    def distribute(self, size=None):
        return np.random.exponential(self.mean, size)


class GODistributionFactory:
    @classmethod
    def parse(cls, parameter_dict: dict):
        keyword = parameter_dict['distribution']
        try:
            method = getattr(cls, f'_parse_{keyword}')
        except AttributeError:
            raise NotImplementedError(f'Distribution {keyword} is invalid')
        return method(parameter_dict)

    def _parse_normal(parameter_dict: dict):
        return GONormalDistribution(parameter_dict)

    def _parse_const(parameter_dict: dict):
        return GOConstDistribution(parameter_dict)

    def _parse_exponential(parameter_dict: dict):
        return GOExponentialDistribution(parameter_dict)


if __name__ == '__main__':
    GODistributionFactory.parse({'distribution': 'normal'})
