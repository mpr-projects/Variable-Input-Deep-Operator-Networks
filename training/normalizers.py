class BaseNormalization:
    def __call__(self, data, *args, **kwargs):
        return self.transform(data, *args, **kwargs)


class ZeroOneNormalization(BaseNormalization):
    """
    Normalizes data to range [0, 1].

    """
    def __init__(self, data):
        self.min = data.min().item()
        self.max = data.max().item()

    def transform(self, data, *args, **kwargs):
        return (data - self.min) / (self.max - self.min)

    def invert(self, data, *args, **kwargs):
        return self.min + (self.max - self.min) * data


class MaxAbsNormalization(BaseNormalization):
    """
    Normalizes data to range [-1, 1]. Actual minima and maxima
    will depend on the data.

    """
    def __init__(self, data):
        self.abs_max = data.abs().max().item()

    def transform(self, data, *args, **kwargs):
        return data / self.abs_max

    def invert(self, data, *args, **kwargs):
        return data * self.abs_max
