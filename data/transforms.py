"""
Definition of problem-specific transform classes
"""

import logging
import numpy as np
from torchvision.transforms import Normalize
from torch import Tensor, squeeze, unsqueeze, round, from_numpy
from data.utils import PhysicalVariables


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
        print("rescale")
        assert True, "RescaleTransform not corrected for new datatype of PhysicalVariables yet"
        # calc max, min of data for each channel of one datapoint, broadcast
        self._data_max, self._data_min = _compute_data_max_and_min(data)
        # self._data_max = self._data_max[:,np.newaxis,np.newaxis,np.newaxis]
        # self._data_min = self._data_min[:,np.newaxis,np.newaxis,np.newaxis]

        # rescale data to new range
        data = data - self._data_min  # normalize to (0, data_max-data_min)
        data /= (self._data_max - self._data_min)  # normalize to (0, 1)
        data *= (self.max - self.min)  # norm to (0, target_max-target_min)
        data += self.min  # normalize to (target_min, target_max)

        return data


def _compute_data_max_and_min(data):
    """
    Calculate the per-channel data maximum and minimum value of a given data point
    :param data: numpy array of shape (Nx)CxHxWxD
        (for N data points with C channels of spatial size HxWxD)
    :returns: per-channels mean and std; numpy array of shape C
    """
    assert True, "RescaleTransform not corrected for new datatype of PhysicalVariables yet"

    # does not check the input type
    for n in data:
        assert type(n) == np.ndarray, "Data is not a numpy array"
    # TODO does this produce/assert the correct output?

    max, min = None, None

    # Calculate the per-channel max and min of the data
    max = np.max(data, axis=(1, 2, 3), keepdims=True)
    min = np.min(data, axis=(1, 2, 3), keepdims=True)

    return max, min


class NormalizeTransform:
    """
    Transform class to normalize data using mean and std
    Functionality depends on the mean and std provided in __init__():
        - if mean and std are single values, normalize the entire image
        - if mean and std are numpy arrays of size C for C image channels,
            then normalize each image channel separately
    """

    def __init__(self, reduced_to_2D: bool = False):
        """
        :param mean: mean of data to be normalized
            can be a single value, or a numpy array of size C
        :param std: standard deviation of data to be normalized
             can be a single value or a numpy array of size C
        """
        self.reduced_to_2D = reduced_to_2D

    def __call__(self, data: PhysicalVariables):
        # manual shift of material IDs  TODO change this in pflotran file!! not here
        names_material = ["Material_ID", "Material ID"]
        for name_material in names_material:
            if name_material in data.keys():
                data[name_material].value -= 1
                mask = data[name_material].value == 2
                data[name_material].value[mask] = -1

        for prop in data.keys():
            # calc mean, std per channel
            data[prop].calc_mean()  # dim=(1, 2, 3), keepdim=True)
            data[prop].calc_std()  # dim=(1, 2, 3), keepdim=True)

        # normalize data to mean and std
        for prop in data.keys():
            Normalize(data[prop].mean_orig, data[prop].std_orig,
                      inplace=True)(data[prop].value)
            # squeeze in case of reduced_to_2D_wrong necessary because unsqueezed before for Normalize to work
            data[prop].value = squeeze(data[prop].value)

        # assert if rounded value of mean is not 0 or of std is not 1
        for prop in data.keys():
            assert round(data[prop].value.mean() * 1e4) == 0, f"Mean of {prop} is not 0 but {data[prop].value.mean()}"
            assert round(data[prop].value.std() * 1e4) == 1e4, f"Std of {prop} is not 1 but {data[prop].value.std()}"
        return data

    def reverse(self, data:PhysicalVariables):
        # reverse normalization
        for prop in data.keys():
            data[prop].value = data[prop].value * \
                data[prop].std_orig + data[prop].mean_orig

        names_material = ["Material_ID", "Material ID"]
        for name_material in names_material:
            if name_material in data.keys():
                mask = data[name_material].value == -1
                data[name_material].value[mask] = 2  # only required, if extraction well (with ID=3) exists
                data[name_material].value += 1
        return data

    def reverse_tensor(self, data:Tensor, **normalize_kwargs) -> Tensor:
        mean_orig, std_orig = normalize_kwargs["mean_orig"], normalize_kwargs["std_orig"]
        # reverse normalization
        data = data * std_orig + mean_orig
        return data


class PowerOfTwoTransform:  # CutOffEdgesTransform:
    """
    Transform class to reduce dimensionality to be a power of 2
    (TODO ?? data cleaning: cut of edges - to get rid of problems with boundary conditions)
        # cut off edges of images with unpretty boundary conditions
        # problem?: "exponential" behaviour at boundaries??
    """

    def __init__(self, oriented="center"):
        self.orientation = oriented

    def __call__(self, data: PhysicalVariables):

        def po2(array, axis):
            dim = array.shape[axis]
            target = 2 ** int(np.log2(dim))
            delta = (dim - target)//2
            if self.orientation == "center":
                result = np.take(array, range(delta, target+delta), axis=axis)
            elif self.orientation == "left":  # cut off only from right
                result = np.take(array, range(0, target), axis=axis)
            return result

        for prop in data.keys():
            for axis in (0, 1, 2):  # ,3): # for axis width, length, depth
                data[prop].value = po2(data[prop].value, axis)

        return data


class ReduceTo2DTransform:
    """
    Transform class to reduce data to 2D, reduce in x, in height of hp: x=7 (if after Normalize)
    """

    def __init__(self, reduce_to_2D_xy=False, loc_hp_x: int = 9):
        # if reduce_to_2D_wrong then the data will still be reduced to 2D but in x,y dimension instead of y,z
        self.reduce_to_2D_xy = reduce_to_2D_xy
        self.loc_hp_x = loc_hp_x

    def __call__(self, data: PhysicalVariables, loc_hp_x:int = None):
        if loc_hp_x is not None:
            self.loc_hp_x = loc_hp_x
            
        # reduce data to 2D
        if self.reduce_to_2D_xy:
            for prop in data.keys():
                data[prop].value.transpose_(0, 2) # (1,3)

        for prop in data.keys():
            assert self.loc_hp_x <= data[prop].value.shape[0], "ReduceTo2DTransform: x is larger than data dimension 0"
            data[prop].value = data[prop].value[self.loc_hp_x, :, :]
            data[prop].value = unsqueeze(data[prop].value, 0)
        logging.info(
            "Reduced data to 2D, but still has dummy dimension 0 for Normalization to work")
        return data

class ToTensorTransform:
    """Transform class to convert np.array-data to torch.Tensor"""

    def __init__(self):
        pass

    def __call__(self, data: PhysicalVariables):
        for prop in data.keys():
            data[prop].value = from_numpy(data[prop].value)
        return data

    def reverse(self, data: PhysicalVariables):
        for prop in data.keys():
            data[prop].value = data[prop].value.numpy()
        return data

    def reverse_tensor(self, data: Tensor, **normalize_kwargs) -> np.ndarray:
        return data.detach() #.numpy() TODO maybe change back if on cpu?


class ComposeTransform:
    """Transform class that combines multiple other transforms into one"""

    def __init__(self, transforms):
        """
        :param transforms: transforms to be combined
        """
        self.transforms = transforms

    def __call__(self, data: PhysicalVariables, loc_hp_x: int = None):
        for transform in self.transforms:
            if isinstance(transform, ReduceTo2DTransform):
                data = transform(data, loc_hp_x)
            else:
                data = transform(data)
        return data

    def reverse(self, data: PhysicalVariables):
        for transform in reversed(self.transforms):
            try:
                data = transform.reverse(data)
            except AttributeError as e:
                pass
                # print(e)
        return data
    
    def reverse_tensor_input(self, data: Tensor, **normalize_kwargs):
        for transform in reversed(self.transforms):
            try:
                data = transform.reverse_tensor(data, **normalize_kwargs)
            except AttributeError as e:
                pass
                # print(e)
        return data