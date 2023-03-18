import numpy as np


class GODistribution:
    def __init__(self) -> None:
        pass

    def sample(self, size=None):
        pass


class GONormalDistribution(GODistribution):
    def __init__(self, parameter_dict: dict) -> None:
        self.mean = parameter_dict['mean']
        self.std = parameter_dict['std']

    def sample(self, size=None):
        if issubclass(type(self.mean), GODistribution):
            self.mean = self.mean.sample()
        if issubclass(type(self.std), GODistribution):
            self.std = self.std.sample()

        return np.random.normal(self.mean, self.std, size)


class GOConstDistribution(GODistribution):
    def __init__(self, parameter_dict: dict) -> None:
        self.const = parameter_dict['const']

    def sample(self, size=None):
        return np.full(size, self.const) if size else self.const


class GOExponentialDistribution(GODistribution):
    def __init__(self, parameter_dict: dict) -> None:
        self.mean = parameter_dict['mean']
        self.offset = parameter_dict['offset'] if 'offset' in parameter_dict else 0

    def sample(self, size=None):
        return np.random.exponential(self.mean - self.offset, size) + self.offset


class GOGammaDistribution(GODistribution):
    def __init__(self, parameter_dict: dict) -> None:
        self.shape = parameter_dict['shape']
        self.scale = parameter_dict['scale']
        self.offset = parameter_dict['offset'] if 'offset' in parameter_dict else 0

    def sample(self, size=None):
        return np.random.gamma(self.shape, self.scale, size) + self.offset

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

    def _parse_gamma(parameter_dict: dict):
        return GOGammaDistribution(parameter_dict)


if __name__ == '__main__':
    GODistributionFactory.parse({'distribution': 'normal'})
