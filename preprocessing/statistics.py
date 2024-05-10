import torch

class WelfordStatistics:
    """
    Track mean and variance of a stream of data using Welford's online algorithm.
    Also track min and max
    The data passed in must be a dict of torch tensors.
    """

    def __init__(self):
        self.__ns = dict()
        self.__means = dict()
        self.__m2s = dict()
        self.__mins = dict()
        self.__maxs = dict()

    def add_data(self, x: dict):
        for key, value in x.items():
            if key not in self.__ns:
                self.__ns[key] = 0
                self.__means[key] = torch.zeros_like(value)
                self.__m2s[key] = 0
                self.__mins[key] = value.min()
                self.__maxs[key] = value.max()
            # use Welford's online algorithm
            self.__ns[key] += 1
            delta = value - self.__means[key]
            self.__means[key] += delta/self.__ns[key]
            self.__m2s[key] += delta*(value - self.__means[key].mean())
            self.__mins[key] = torch.min(self.__mins[key], value.min())
            self.__maxs[key] = torch.max(self.__maxs[key], value.max())

    def mean(self):
        result = dict()
        for key in self.__ns:
            result[key] = self.__means[key].mean().item()
        return result

    def var(self):
        result = dict()
        for key in self.__ns:
            if self.__ns[key] < 2:
                result[key] = 0
            else:
                result[key] = (self.__m2s[key]/(self.__ns[key]-1)).mean()
        return result

    def std(self):
        result = dict()
        for key in self.__ns:
            result[key] = (torch.sqrt(self.var()[key])).item()
        return result
    
    def min(self):
        result = dict()
        for key in self.__ns:
            result[key] = self.__mins[key].item()
        return result
    
    def max(self):
        result = dict()
        for key in self.__ns:
            result[key] = self.__maxs[key].item()
        return result