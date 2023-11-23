"""
Definition of problem-specific transform classes
"""

import logging
from typing import Tuple

import numpy as np
import torch
from torch import linalg, nonzero, unsqueeze


class NormalizeTransform:
    def __init__(self,info:dict,out_range = (0,1)):
        self.info = info
        self.input_stats = self.info["Inputs"]
        self.label_stats = self.info["Labels"]
        self.out_min, self.out_max = out_range 

    def __call__(self,data, type = "Inputs"):
        for prop, stats in self.info[type].items():
            index = stats["index"]
            if index < data.shape[0]:
                self.__apply_norm(data,index,stats)
            else:
                logging.warning(f"Index {index} might be in training data but not in this dataset")
        return data
    
    def reverse(self,data,type = "Labels"):
        for prop, stats in self.info[type].items():
            index = stats["index"]
            self.__reverse_norm(data,index,stats)
        return data
    
    def __apply_norm(self,data,index,stats):
        norm = stats["norm"]
        if norm == "Rescale":
            delta = stats["max"] - stats["min"]
            data[index] = (data[index] - stats["min"]) / delta * (self.out_max - self.out_min) + self.out_min
        elif norm == "Standardize":
            data[index] = (data[index] - stats["mean"]) / stats["std"]
        elif norm is None:
            pass
        else:
            raise ValueError(f"Normalization type '{stats['Norm']}' not recognized")
        
    def __reverse_norm(self,data,index,stats):
        if len(data.shape) == 4:
            assert data.shape[0] < data.shape[1], "Properties must be in 0th dimension; batches pushed to 1st dimension"
        norm = stats["norm"]
        if norm == "Rescale":
            delta = stats["max"] - stats["min"]
            data[index] = (data[index] - self.out_min) / (self.out_max - self.out_min) * delta + stats["min"]
        elif norm == "Standardize":
            data[index] = data[index] * stats["std"] + stats["mean"]
        elif norm is None:
            pass
        else:
            raise ValueError(f"Normalization type '{stats['Norm']}' not recognized")

class SignedDistanceTransform:
    """
    Transform class to calculate signed distance transform for material id.  
    This transform takes a dict of tensors as input and returns a dict of tensors.
    """

    def __init__(self):
        pass

    def __call__(self, data: dict):
        logging.info("Start SignedDistanceTransform")

        # check if SDF is in data (inputs vs. labels)
        if "SDF" not in data.keys():
            logging.info("No material ID in data, no SignedDistanceTransform")
            return data

        def get_loc_hp():
            if "SDF" in data.keys():
                loc_hp = nonzero(data["SDF"] == torch.max(
                    data["SDF"])).squeeze()
                if len(loc_hp) != 3:
                    logging.info(f"loc_hp returns more than one position: {loc_hp}")
                return loc_hp

        data["SDF"] = self.sdf(data["SDF"].float(), get_loc_hp().float())
        logging.info("SignedDistanceTransform done")
        return data
    
    def sdf(self, data: torch.tensor, loc_hp: torch.tensor):
        for index_x in range(data.shape[0]):
            for index_y in range(data.shape[1]):
                try:
                    for index_z in range(data.shape[2]):
                        data[index_x, index_y, index_z] = linalg.norm(
                            torch.tensor([index_x, index_y, index_z]).float() - loc_hp)
                except IndexError:
                    data[index_x, index_y] = linalg.norm(
                        torch.tensor([index_x, index_y]).float() - loc_hp)

        data = 1 - data / data.max()
        return data


class MultiHPDistanceTransform:
    """
    Transform class to calculate distance transform for multiple heat pumps for material id.  
    This transform takes a dict of tensors as input and returns a dict of tensors.
    """

    def __init__(self):
        pass

    def __call__(self, data: dict):
        logging.info("Start MultiHPDistanceTransform")

        # check if MDF is in data (inputs vs. labels)
        if "MDF" not in data.keys():
            logging.info("No material ID in data, no MultiHPDistanceTransform")
            return data

        positions = torch.tensor(np.array(np.where(data["Material ID"] == torch.max(data["Material ID"])))).T
        assert positions.dim() == 2, "MDF + just 1 HP? u sure?"
        data["MDF"] = self.mdf(data["MDF"], positions)
        logging.info("MultiHPDistanceTransform done")

        return data
    
    def mdf(self, data: torch.tensor, positions: torch.tensor):
        assert (data == 0).all(), "all data should be 0"
        max_dist = 50 # max_dist between 2 points / manual influence distance : also for scaling - TODO: optimize value

        for dist in range(int(max_dist)):
            for x in range(dist):
                for y in range(dist):
                    for point in positions:
                        if np.sqrt(x**2+y**2) <= dist:
                            point2 = point + torch.tensor([x,y,0])
                            try:
                                if data[point2[0], point2[1]] == 0:
                                    data[point2[0], point2[1]] = (max_dist - dist)/max_dist
                            except: # outside domain
                                pass
                            try:
                                point2 = point + torch.tensor([-x,y,0])
                                if data[point2[0], point2[1]] == 0:
                                    data[point2[0], point2[1]] = (max_dist - dist)/max_dist
                            except: # outside domain
                                pass
                            try:
                                point2 = point + torch.tensor([x,-y,0])
                                if data[point2[0], point2[1]] == 0:
                                    data[point2[0], point2[1]] = (max_dist - dist)/max_dist
                            except: # outside domain
                                pass
                            try:
                                point2 = point + torch.tensor([-x,-y,0])
                                if data[point2[0], point2[1]] == 0:
                                    data[point2[0], point2[1]] = (max_dist - dist)/max_dist
                            except: # outside domain
                                pass
        return data


class PowerOfTwoTransform:
    """
    Transform class to reduce dimensionality to be a power of 2
    (TODO ?? data cleaning: cut of edges - to get rid of problems with boundary conditions)
        # cut off edges of images with unpretty boundary conditions
        # problem?: "exponential" behaviour at boundaries??
    """

    def __init__(self, oriented="left"): # TODO
        self.orientation = oriented

    def __call__(self, data):
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
                data[prop] = po2(data[prop], axis)

        logging.info("Data reduced to power of 2")
        return data


class ReduceTo2DTransform:
    """
    Transform class to reduce data to 2D, reduce in x, in height of hp: x=7
    This Transform takes a dict of tensors as input and returns a dict of tensors
    """

    def __init__(self, reduce_to_2D_xy=False):
        # if reduce_to_2D_wrong then the data will still be reduced to 2D but in x,y dimension instead of y,z
        self.reduce_to_2D_xy = reduce_to_2D_xy
        if reduce_to_2D_xy:
            self.slice_dimension = 2  # z
        else:
            self.slice_dimension = 0  # x
        loc_hp = np.array([2, 9, 2])
        self.loc_hp_slice = int(loc_hp[self.slice_dimension])

    def __call__(self, data, loc_hp: Tuple = None):
        logging.info("Start ReduceTo2DTransform")
        already_2d: bool = False

        for data_prop in data.keys():
            # check if data is already 2D, if so: do nothing/ only switch axes (for plotting)
            data_shape = data[data_prop].shape
            if 1 in data_shape or len(data_shape) == 2:
                if data_shape[-1] == 1 and len(data_shape) == 3:
                    data[data_prop] = np.swapaxes(
                        data[data_prop], 0, 1)
                already_2d = True

            # else: reduce data to 2D
        if not already_2d:
            if loc_hp is not None:
                self.loc_hp_slice = loc_hp[self.slice_dimension]

            # else: reduce data to 2D
            if self.reduce_to_2D_xy:
                for prop in data.keys():
                    data[prop].transpose_(0, 2)  # (1,3)
            for prop in data.keys():
                assert self.loc_hp_slice <= data[prop].shape[
                    0], "ReduceTo2DTransform: x is larger than data dimension 0"
                data[prop] = data[prop][self.loc_hp_slice, :, :]
                data[prop] = unsqueeze(
                    data[prop], 0)  # TODO necessary?
        logging.info(
            "Reduced data to 2D, but still has dummy dimension 0 for Normalization to work")
        return data


class ComposeTransform:
    """Transform class that combines multiple other transforms into one"""

    def __init__(self, transforms):
        """
        :param transforms: transforms to be combined
        """
        self.transforms = transforms

    def __call__(self, data, loc_hp: Tuple = None):
        for transform in self.transforms:
            if isinstance(transform, ReduceTo2DTransform):
                data = transform(data, loc_hp)
            else:
                data = transform(data)
        return data

    def reverse(self, data, **normalize_kwargs):
        for transform in reversed(self.transforms):
            try:
                data = transform.reverse(data, **normalize_kwargs)
            except AttributeError as e:
                pass
        return data


class ToTensorTransform:
    """Transform class to convert dict of tensors to one tensor"""

    def __init__(self):
        pass

    def __call__(self, data: dict):
        logging.info("Start ToTensorTransform")
        result: torch.Tensor = None
        for prop in data.keys():
            if result is None:
                result = data[prop].squeeze()[None, ...]
            else:
                result = torch.cat(
                    (result, data[prop].squeeze()[None, ...]), axis=0)
        logging.info("Converted data to torch.Tensor")
        return result


if __name__ == "__main__":
    trafo = ComposeTransform([SignedDistanceTransform()])
    data = {"SDF": torch.zeros((9, 9, 2))}
    data["SDF"][1,1,0] = 1
    print(data["SDF"][:,:,0])
    data = trafo(data)
    print(data["SDF"])