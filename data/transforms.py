"""
Definition of problem-specific transform classes
"""

import numpy as np
from torchvision.transforms import Normalize
import torch

def compute_data_max_and_min(data):
    """
    Calculate the per-channel data maximum and minimum value of a given data point
    :param data: numpy array of shape (Nx)CxHxWxD
        (for N data points with C channels of spatial size HxWxD)
    :returns: per-channels mean and std; numpy array of shape C
    """
    max, min = None, None

    # Calculate the per-channel max and min of the data
    max = np.max(data, axis=(1,2,3), keepdims=True)
    min = np.min(data, axis=(1,2,3), keepdims=True)

    return max, min
    
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

    def __call__(self, data, **kwargs):
        # calc max, min of data for each channel of one datapoint, broadcast
        self._data_max, self._data_min = compute_data_max_and_min(data)
        # self._data_max = self._data_max[:,np.newaxis,np.newaxis,np.newaxis]
        # self._data_min = self._data_min[:,np.newaxis,np.newaxis,np.newaxis]

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

    def __init__(self, reduced_to_2D=False):
        """
        :param mean: mean of data to be normalized
            can be a single value, or a numpy array of size C
        :param std: standard deviation of data to be normalized
             can be a single value or a numpy array of size C
        """
        self.mean = None
        self.std = None
        self.reduced_to_2D = reduced_to_2D

    def __call__(self, data, index_material_id=None):
        # manual shift of material IDs  TODO change this in pflotran file!! not here
        if self.reduced_to_2D:
            if index_material_id:
                data[index_material_id,:,:] -= 1
                # only required, if extraction well (with ID=3) exists
                mask = data[index_material_id,:,:] == 2
                data[index_material_id,:,:][mask] = -1
        else:
            if index_material_id:
                data[index_material_id,:,:,:] -= 1
                # only required, if extraction well (with ID=3) exists
                mask = data[index_material_id,:,:,:] == 2
                data[index_material_id,:,:,:][mask] = -1

        bool_just_one_input=False
        # ugly workaround for just one input variable
        if data.shape[0] == 1: # if only one input variable like Material_ID
            bool_just_one_input = True
            temp_zeros = torch.zeros(data.shape)
            data = torch.cat([data, temp_zeros], dim=0)

        if self.reduced_to_2D:
            # calc mean, std per channel then normalize data to mean and std, including broadcasting
            self.mean = data.mean(dim=(1, 2), keepdim=True)
            self.std = data.std(dim=(1, 2), keepdim=True)
        else:
            # calc mean, std per channel then normalize data to mean and std, including broadcasting
            self.mean = data.mean(dim=(1, 2, 3), keepdim=True)
            self.std = data.std(dim=(1, 2, 3), keepdim=True)
        
        # ugly workaround for just one input variable
        if bool_just_one_input:
            self.mean = torch.tensor_split(self.mean, 2)[0]
            self.std = torch.tensor_split(self.std, 2)[0]
            data = torch.tensor_split(data, 2)[0]
        
        data = Normalize(self.mean, self.std, inplace=True)(data)
        # assert if rounded value of mean is not 0

        assert torch.abs(torch.round(data.mean(dim=(0,1,2)),decimals=12)) == 0
        assert torch.abs(torch.round(data.std(dim=(0,1,2)),decimals=12)) <= 1
        assert torch.abs(torch.round(data.std(dim=(0,1,2)),decimals=12)) >= 0.999
        return data

    def reverse(self, data, mean=None, std=None, index_material_id=None):
        # reverse normalization
        # data = torch.from_numpy(data) * self.std + self.mean # version if std, mean in class
        data = torch.add(torch.mul(data,std), mean)
        #TODO why is data numpy format??
        
        # manual shift of material IDs  TODO change this in pflotran file!! not here
        if self.reduced_to_2D:
            if index_material_id:
                mask = data[index_material_id,:,:] == -1

                # only required, if extraction well (with ID=3) exists
                data[index_material_id,:,:][mask] = 2
                data[index_material_id,:,:] += 1
        else:
            if index_material_id:
                mask = data[index_material_id,:,:,:] == -1

                # only required, if extraction well (with ID=3) exists
                data[index_material_id,:,:,:][mask] = 2
                data[index_material_id,:,:,:] += 1

        return data #, data.mean(dim=(1, 2, 3), keepdim=True), data.std(dim=(1, 2, 3), keepdim=True)

class PowerOfTwoTransform: #CutOffEdgesTransform:
    """
    Transform class to reduce dimensionality to be a power of 2
    (TODO ?? data cleaning: cut of edges - to get rid of problems with boundary conditions)
        # cut off edges of images with unpretty boundary conditions
        # problem?: "exponential" behaviour at boundaries??
    """
    def __init__(self, oriented="center"):
        self.orientation = oriented

    def __call__(self, data, **kwargs):
        
        def po2(array, axis):
            dim = array.shape[axis]
            target = 2 ** int(np.log2(dim))
            delta = (dim - target)//2
            if self.orientation == "center":
                result = np.take(array, range(delta, target+delta), axis=axis)
            elif self.orientation == "left": # cut off only from right
                result = np.take(array, range(0, target), axis=axis)
            return result
        
        for axis in (1,2,3): # for axis width, length, depth
            data = po2(data, axis)

        return data
        #return data_np[:,1:-1,1:-3,1:-1]

class ReduceTo2DTransform:
    """
    Transform class to reduce data to 2D, reduce in x, in height of hp: x=7 (if after Normalize)
    #TODO still np?
    """
    def __init__(self, reduce_to_2D_wrong=False):
        # if reduce_to_2D_wrong then the data will still be reduced to 2D but in x,y dimension instead of y,z
        self.reduce_to_2D_wrong = reduce_to_2D_wrong

    def __call__(self, data, x=9, **kwargs):
        # reduce data to 2D
        if self.reduce_to_2D_wrong:
            x = 9
            data.transpose_(1,3)
        
        return data[:,x,:,:]

class ToTensorTransform:
    """Transform class to convert np.array-data to torch.Tensor"""
    def __init__(self):
        pass
    def __call__(self, data, **kwargs):
        return torch.from_numpy(data)

class ComposeTransform:
    """Transform class that combines multiple other transforms into one"""
    def __init__(self, transforms):
        """
        :param transforms: transforms to be combined
        """
        self.transforms = transforms
        self.mean = None
        self.std = None

    def __call__(self, data, **kwargs):
        for transform in self.transforms:
            data = transform(data, **kwargs)
            try:
                self.mean = transform.mean
                self.std = transform.std
            except:
                pass
        return data , self.mean, self.std

    def reverse(self, data, **kwargs):
        for transform in reversed(self.transforms):
            try:
                data = transform.reverse(data, **kwargs)
            except AttributeError:
                pass
                #print(f"for transform {transform} no reverse implemented")
        return data
