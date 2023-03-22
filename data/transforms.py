"""
Definition of problem-specific transform classes
"""

import logging
import numpy as np
from torchvision.transforms import Normalize
from torch import Tensor, squeeze, unsqueeze, round, from_numpy, linalg, nonzero
import torch
from data.utils import PhysicalVariables
from typing import Dict, Union, Tuple


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

    def __init__(self, sdf_bool:bool=True):
        """
        :param mean: mean of data to be normalized
            can be a single value, or a numpy array of size C
        :param std: standard deviation of data to be normalized
             can be a single value or a numpy array of size C
        """
        self.name_material = "Material ID"
        self.sdf_bool:bool= sdf_bool

    def __call__(self, data: PhysicalVariables, mean_val:Dict, std_val:Dict):
        logging.info("Start normalization")
        if not self.sdf_bool:
            # index shift for material ID
            if self.name_material in data.keys():
                data[self.name_material].value -= 1
                mask = data[self.name_material].value == 2
                data[self.name_material].value[mask] = -1
                data[self.name_material].value = data[self.name_material].value.double() # casting from int to double

        # normalize data to mean and std
        for prop in data.keys():
            if not self.sdf_bool:
                Normalize(mean_val[prop], std_val[prop], inplace=True)(data[prop].value)
            else:
                if prop != self.name_material:
                    Normalize(mean_val[prop], std_val[prop], inplace=True)(data[prop].value)
            # # squeeze in case of reduced_to_2D_wrong necessary because unsqueezed before for Normalize to work
            data[prop].value = squeeze(data[prop].value)
        logging.info("Data normalized")
        return data

    def init_mean_std(self, data: PhysicalVariables):
        if not self.sdf_bool:
            if self.name_material in data.keys():
                data[self.name_material].value -= 1
                mask = data[self.name_material].value == 2
                data[self.name_material].value[mask] = -1

        for prop in data.keys():
            if not self.sdf_bool:
                # calc mean, std per channel
                data[prop].calc_mean()  # dim=(1, 2, 3), keepdim=True)
                data[prop].calc_std()  # dim=(1, 2, 3), keepdim=True)
            else:
                if prop != self.name_material:
                    # calc mean, std per channel
                    data[prop].calc_mean()  # dim=(1, 2, 3), keepdim=True)
                    data[prop].calc_std()  # dim=(1, 2, 3), keepdim=True)

        return data

    def reverse(self, data:PhysicalVariables, mean_val:Dict, std_val:Dict):
        # reverse normalization
        if not self.sdf_bool:
            for prop in data.keys():
                data[prop].value = data[prop].value * std_val[prop] + mean_val[prop]
        else:
            for prop in data.keys():
                if prop not in self.name_material:
                    data[prop].value = data[prop].value * std_val[prop] + mean_val[prop]

        if not self.sdf_bool:
            for name_material in self.name_material:
                if name_material in data.keys():
                    mask = data[name_material].value == -1
                    data[name_material].value[mask] = 2  # only required, if extraction well (with ID=3) exists
                    data[name_material].value += 1
        return data

    def reverse_tensor(self, data:Tensor, mean_val, std_val) -> Tensor:
        # reverse normalization
        data = data * std_val + mean_val
        # TODO exclude Material ID???
        return data

class SignedDistanceTransform:
    """
    Transform class to calculate signed distance transform
    """

    def __init__(self):
        self.name_material = "Material ID"

    def __call__(self, data: PhysicalVariables):
        logging.info("Start SignedDistanceTransform")

        # check if material ID is in data (inputs vs. labels)
        material_id_in_data = False
        if self.name_material in data.keys():
            material_id_in_data = True
        if not material_id_in_data:
            logging.info("No material ID in data, no SignedDistanceTransform")
            return data

        def get_loc_hp():
            if self.name_material in data.keys():
                loc_hp = nonzero(data[self.name_material].value==torch.max(data[self.name_material].value)).squeeze()
                return loc_hp
            
        loc_hp = get_loc_hp()
        data[self.name_material].value = data[self.name_material].value.float()
        for index_x in range(data[self.name_material].value.shape[0]):
            for index_y in range(data[self.name_material].value.shape[1]):
                for index_z in range(data[self.name_material].value.shape[2]):
                    data[self.name_material].value[index_x, index_y, index_z] = linalg.norm(
                        torch.tensor([index_x, index_y, index_z]).float() - loc_hp.float())

        data[self.name_material].value = data[self.name_material].value / torch.max(data[self.name_material].value)

        return data
    
    ## NOT TESTED AND NOT CALLED
    # def reverse(self, data: PhysicalVariables):
    #     logging.info("Start reverse SignedDistanceTransform")
    #     # check if material ID is in data (inputs vs. labels)
    #     material_id_in_data = False
    #     for name_material in self.names_material:
    #         if name_material in data.keys():
    #             material_id_in_data = True
    #     if not material_id_in_data:
    #         logging.info("No material ID in data, no reverse SignedDistanceTransform")
    #         return data

    #     def get_loc_hp():
    #         for name_material in self.names_material:
    #             if name_material in data.keys():
    #                 loc_hp = nonzero(data[name_material].value==torch.min(data[name_material].value)).squeeze()
    #                 return loc_hp, name_material
    #     loc_hp, name_material = get_loc_hp()

    #     # zeros tensor of size of data[name_material].value
    #     data[name_material].value = torch.zeros(data[name_material].value.shape)
    #     data[name_material].value[loc_hp[0], loc_hp[1], loc_hp[2]] = 1

    #     return data

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
        logging.info("Start PowerOfTwoTransform")
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

        logging.info("Data reduced to power of 2")
        return data


class ReduceTo2DTransform:
    """
    Transform class to reduce data to 2D, reduce in x, in height of hp: x=7 (if after Normalize)
    """

    def __init__(self, reduce_to_2D_xy=False):
        # if reduce_to_2D_wrong then the data will still be reduced to 2D but in x,y dimension instead of y,z
        self.reduce_to_2D_xy = reduce_to_2D_xy
        if reduce_to_2D_xy:
            self.slice_dimension = 2 # z
        else:
            self.slice_dimension = 0 # x
        loc_hp = np.array([2,9,2])
        self.loc_hp_slice = int(loc_hp[self.slice_dimension])

    def __call__(self, data: PhysicalVariables, loc_hp: Tuple=None):
        logging.info("Start ReduceTo2DTransform")
        already_2d:bool = False

        for data_prop in data.keys():
            # check if data is already 2D, if so: do nothing/ only switch axes (for plotting)
            data_shape = data[data_prop].value.shape
            if 1 in data_shape or len(data_shape) == 2:
                if data_shape[-1] == 1 and len(data_shape) == 3:
                    data[data_prop].value = np.swapaxes(data[data_prop].value, 0, 1)
                already_2d = True

            # else: reduce data to 2D
        if not already_2d:
            if loc_hp is not None:
                self.loc_hp_slice = loc_hp[self.slice_dimension]

            #else: reduce data to 2D
            if self.reduce_to_2D_xy:
                for prop in data.keys():
                    data[prop].value.transpose_(0, 2) # (1,3)
            for prop in data.keys():
                assert self.loc_hp_slice <= data[prop].value.shape[0], "ReduceTo2DTransform: x is larger than data dimension 0"
                data[prop].value = data[prop].value[self.loc_hp_slice, :, :]
                data[prop].value = unsqueeze(data[prop].value, 0) # TODO necessary?

        logging.info("Reduced data to 2D, but still has dummy dimension 0 for Normalization to work")
        return data


class ToTensorTransform:
    """Transform class to convert np.array-data to torch.Tensor"""

    def __init__(self):
        pass

    def __call__(self, data: PhysicalVariables):
        logging.info("Start ToTensorTransform")
        for prop in data.keys():
            data[prop].value = from_numpy(data[prop].value)
        logging.info("Converted data to torch.Tensor")
        return data

    ##TODO include when change back to cpu?
    # def reverse(self, data: PhysicalVariables, **normalize_kwargs):
    #     for prop in data.keys():
    #         data[prop].value = data[prop].value.numpy()
    #     return data

    def reverse_tensor(self, data: Tensor, **normalize_kwargs) -> np.ndarray:
        return data.detach() #.numpy()


class ComposeTransform:
    """Transform class that combines multiple other transforms into one"""

    def __init__(self, transforms):
        """
        :param transforms: transforms to be combined
        """
        self.transforms = transforms

    def __call__(self, data: PhysicalVariables, loc_hp:Tuple=None, mean_val:Dict=None, std_val:Dict=None):
        for transform in self.transforms:
            if isinstance(transform, ReduceTo2DTransform):
                data = transform(data, loc_hp)
            elif isinstance(transform, NormalizeTransform):
                if not mean_val==None and not std_val==None:
                    data = transform(data, mean_val, std_val)
                else:
                    data = transform.init_mean_std(data)
            else:
                data = transform(data)
        return data

    def reverse(self, data: PhysicalVariables, **normalize_kwargs):
        for transform in reversed(self.transforms):
            try:
                data = transform.reverse(data, **normalize_kwargs)
            except AttributeError as e:
                pass
        return data

    def reverse_tensor_input(self, data: Tensor, **normalize_kwargs):
        for transform in reversed(self.transforms):
            try:
                data = transform.reverse_tensor(data, **normalize_kwargs)
            except AttributeError as e:
                pass
        return data
