"""
Definition of problem-specific transform classes
"""

import numpy as np

def compute_data_max_and_min(data):
    """
    Calculate the per-channel data maximum and minimum value of a given data point
    :param data: numpy array of shape (Nx)CxHxWxD
        (for N data points with C channels of spatial size HxWxD)
    :returns: per-channels mean and std; numpy array of shape C
    """
    max, min = None, None

    # Calculate the per-channel max and min of the data  #
    max = np.max(data, axis=(1,2,3))
    min = np.min(data, axis=(1,2,3))

    return max, min

def compute_image_mean_and_std(data):
    """
    Calculate the per-channel mean and standard deviation of given data
    :param data: numpy array of shape (Nx)CxHxWxD
        (for N data points with C channels of spatial size HxWxD)
    :returns: per-channels mean and std; numpy array of shape C
    """
    mean, std = None, None

    mean = np.mean(data, axis=(1, 2, 3))
    std = np.std(data, axis=(1, 2, 3))

    return mean, std

class RescaleTransform:
    """Transform class to rescale data to a given range"""
    def __init__(self, out_range=(0, 1)):
        """
        :param out_range: Value range to which data should be rescaled to
        :param in_range:  Old value range of the data
            e.g. (0, 255) for images with raw pixel values
        """
        self.min = out_range[0]
        self.max = out_range[1]

    def __call__(self, data):
        # calc max, min of data for each channel of one datapoint, broadcast
        self._data_max, self._data_min = compute_data_max_and_min(data)
        self._data_max = self._data_max[:,np.newaxis,np.newaxis,np.newaxis]
        self._data_min = self._data_min[:,np.newaxis,np.newaxis,np.newaxis]

        # rescale data to new range
        data = data - self._data_min  # normalize to (0, data_max-data_min)
        data /= (self._data_max - self._data_min)  # normalize to (0, 1)
        data *= (self.max - self.min)  # norm to (0, target_max-target_min)
        data += self.min  # normalize to (target_min, target_max)

        return data

class NormalizeTransform:
    """
    Transform class to normalize data using mean and std
    Functionality depends on the mean and std provided in __init__():
        - if mean and std are single values, normalize the entire image
        - if mean and std are numpy arrays of size C for C image channels,
            then normalize each image channel separately
    """

    def __init__(self):
        """
        :param mean: mean of data to be normalized
            can be a single value, or a numpy array of size C
        :param std: standard deviation of data to be normalized
             can be a single value or a numpy array of size C
        """
        self.mean = None
        self.std = None

    def __call__(self, data):
        # normalize data to mean and std
        self.mean, self.std = compute_image_mean_and_std(data)
        self.mean = self.mean[:,np.newaxis,np.newaxis,np.newaxis]
        self.std = self.std[:,np.newaxis,np.newaxis,np.newaxis]

        if isinstance(self.mean, np.ndarray): # checks whether mean is an array
            data = data - self.mean
        else:
            data = data - self.mean * np.ones(data.shape)
        if isinstance(self.std, np.ndarray):
            data = data / self.std
        else:
            data = data / self.std * np.ones(data.shape)

        return data

' TODO: implement flatten transform'


class PowerOfTwoTransform: #CutOffEdgesTransform:
    """
    Transform class to reduce dimensionality to be a power of 2
    (TODO ?? data cleaning: cut of edges - to get rid of problems with boundary conditions)
    """
    def __init__(self):
        ...

    def __call__(self, data_np):
        # cut off edges of images with unpretty boundary conditions
        # no problem? just exponential behaviour at boundaries??
            
        def po2(array, axis):
            dim = array.shape[axis]
            target = 2 ** int(np.log2(dim))
            delta = (dim - target)//2

            return np.take(array, range(delta, target+delta), axis=axis)
        for axis in (1,2,3): # for axis width, length, depth
            data_np = po2(data_np, axis)
        return data_np
        #return data_np[:,1:-1,1:-3,1:-1]

class ReduceTo2DTransform:
    """
    Transform class to reduce data to 2D, reduce in x, in height of hp: x=8
    """
    def __init__(self):
        pass

    def __call__(self, data_np, x=8):
        # reduce data to 2D
        return data_np[:,x,:,:]

class ComposeTransform:
    """Transform class that combines multiple other transforms into one"""
    def __init__(self, transforms):
        """
        :param transforms: transforms to be combined
        """
        self.transforms = transforms

    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data