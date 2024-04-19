"""
Definition of problem-specific transform classes
"""

import logging
from typing import Tuple
from pathlib import Path
from tqdm.auto import tqdm

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import linalg, nonzero, unsqueeze


class NormalizeTransform:
    def __init__(self,info:dict,out_range = (0,1)):
        self.info = info
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
        
        def rescale():
            delta = stats["max"] - stats["min"]
            data[index] = (data[index] - stats["min"]) / delta * (self.out_max - self.out_min) + self.out_min
        
        if norm == "LogRescale":
            data[index] = np.log(data[index] - stats["min"] + 1)
            rescale()
        elif norm == "Rescale":
            rescale()
        elif norm == "Standardize":
            data[index] = (data[index] - stats["mean"]) / stats["std"]
        elif norm is None:
            pass
        else:
            raise ValueError(f"Normalization type '{stats['norm']}' not recognized")
        
    def __reverse_norm(self,data,index,stats):
        if len(data.shape) == 4:
            assert data.shape[0] <= data.shape[1], "Properties must be in 0th dimension; batches pushed to 1st dimension"
        norm = stats["norm"]

        def rescale():
            delta = stats["max"] - stats["min"]
            data[index] = (data[index] - self.out_min) / (self.out_max - self.out_min) * delta + stats["min"]

        if norm == "LogRescale":
            rescale()
            data[index] = np.exp(data[index]) + stats["min"] - 1
        elif norm == "Rescale":
            rescale()
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
    
class PositionalEncodingTransform:
    """
    Transform class to add positional encoding to the data.
    This transform takes a dict of tensors as input and returns a dict of tensors.
    
    The positional encoding is added to the inputs.
    
    The positional encoding is calculated as follows:
    - The position of the highest value in the material id is determined.
    - The distance of each voxel to the position is calculated.
    - The distance is normalized.
    - The distance is added to the inputs.
    """

    def __init__(self):
        pass

    def __call__(self, data: dict):
        logging.info("Start PositionalEncodingTransform")

        # assert if only PE x or PE y is in data
        assert "PE x" in data.keys() and "PE y" in data.keys() or "PE x" not in data.keys() and "PE y" not in data.keys(), "Only one of PE x and PE y is in data"

        # check if SDF is in data (inputs vs. labels)
        if "PE x" not in data.keys():
            logging.info("No material ID in data, no PositionalEncodingTransform")
            return data

        def get_loc_hp(data: torch.tensor):
            loc_hp = nonzero(data == torch.max(data)).squeeze()
            if len(loc_hp) != 3:
                logging.warning(f"loc_hp returns more than one position: {loc_hp}")
            return loc_hp
        loc_hp = get_loc_hp(data["PE x"])

        data["PE x"] = self.pe_x(data["PE x"], loc_hp.float()[0])
        data["PE y"] = self.pe_y(data["PE y"], loc_hp.float()[1])
        logging.info("PositionalEncodingTransform done")
        return data

    def pe_x(self, data: torch.tensor, loc_hp: torch.tensor):
        data_shape = data.shape
        data = torch.zeros(data_shape)
        # fill data with idx in x direction
        for index_x in range(data.shape[0]):
            data[index_x, :] = index_x
        data -= loc_hp
        data = data.abs()
        data /= data.max() #256 
        data = 1 - data
        # data[data < 0] = 0
        assert data.shape == data_shape, f"PE x has wrong shape: {data.shape} instead of {data_shape}"
        return data
    
    def pe_y(self, data: torch.tensor, loc_hp: torch.tensor):
        data_shape = data.shape
        data = torch.zeros(data_shape)
        # fill data with idx in y direction
        for index_y in range(data.shape[1]):
            data[:, index_y] = index_y
        data -= loc_hp
        data = data.abs()
        data /= data.max() # 16 #
        data = 1 - data
        # data[data < 0] = 0
        assert data.shape == data_shape, f"PE y has wrong shape: {data.shape} instead of {data_shape}"
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

        positions = torch.tensor(np.array(np.where(data["MDF"] == torch.max(data["MDF"])))).T
        assert positions.dim() == 2, "MDF + just 1 HP? u sure?"
        data["MDF"] = self.mdf(data["MDF"], positions)

        logging.info("MultiHPDistanceTransform done")
        return data
    
    def mdf(self, data: torch.tensor, positions: torch.tensor):
        max_dist = 100 # max_dist between 2 points / manual influence distance : also for scaling - TODO: optimize value
        kdtree = cKDTree(positions[:,:2])
        data_shape = data.shape
        X, Y = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]))
        data_tmp, _ = kdtree.query(np.array([X, Y]).T.reshape(-1, 2), k=1)
        data = np.minimum(data_tmp, max_dist)
        data = data.reshape(data_shape)
        data = 1 - data / np.max(data)
        data = torch.tensor(data, dtype=torch.float32)
        return data
    

class LinearSmearTransform:
    """
    Transform class to calculate distance transform for multiple heat pumps for material id - just depending on the next hp upstream.
    This transform takes a dict of tensors as input and returns a dict of tensors.
    """

    def __init__(self):
        pass

    def __call__(self, data: dict):
        logging.info("Start LinSmearTrafo")

        # check if LST is in data (inputs vs. labels)
        if "LST" not in data.keys():
            logging.info("No material ID in data, no LinSmearTrafo")
            return data

        assert len(data["LST"].size()) == 3, "wrong number of dimensions for LinSmearTrafo, expect 4"
        data["LST"] = self.lin_smear(data["LST"])
        logging.info("LinSmearTrafo done")

        return data
    
    def lin_smear(self, data: torch.tensor):
        max_dist = int(data.shape[0] * 1.1)
        
        for idx in range(1, data.shape[0]):
            data[idx] = np.maximum(data[idx-1]*(1-1/max_dist), data[idx])
        
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
                data[prop] = unsqueeze(data[prop], 0)  # TODO necessary?
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


def get_transforms(reduce_to_2D: bool = True, reduce_to_2D_xy: bool = True, power2trafo: bool = False, problem:str="2stages"):
    transforms_list = []

    if reduce_to_2D:
        transforms_list.append(ReduceTo2DTransform(
            reduce_to_2D_xy=reduce_to_2D_xy))
    if power2trafo:
        transforms_list.append(PowerOfTwoTransform())
    transforms_list.append(SignedDistanceTransform())
    if problem in ["extend1", "extend2"]:
        transforms_list.append(PositionalEncodingTransform())
    elif problem == "allin1":
        transforms_list.append(MultiHPDistanceTransform())
        transforms_list.append(LinearSmearTransform())

    transforms = ComposeTransform(transforms_list)
    return transforms

def normalize(dataset_path: Path, info: dict, total: int = None):
    """
    Apply the normalization using the stats from `info` to the dataset in `dataset_path`.

    Parameters
    ----------
        dataset_path : str
            Path to the dataset to normalize.
        info : dict
            Dictionary containing the normalization stats:  
            {  
                inputs: {"key": {"mean": float, "std": float, "index": int}},  
                labels: {"key": {"mean": float, "std": float, "index": int}}  
            }
        total : int
            Total number of files to normalize. Used for tqdm progress bar.

    """
    norm = NormalizeTransform(info)
    for input_file in tqdm((dataset_path / "Inputs").iterdir(), desc="Normalizing inputs", total=total):
        x = torch.load(input_file)
        x = norm(x,"Inputs")
        torch.save(x, input_file)
    for label_file in tqdm((dataset_path / "Labels").iterdir(), desc="Normalizing labels", total=total):
        y = torch.load(label_file)
        y = norm(y,"Labels")
        torch.save(y, label_file)


if __name__ == "__main__":
    trafo = ComposeTransform([PositionalEncodingTransform()])
    data = {"PE x": torch.zeros((5, 6, 2)),
            "PE y": torch.zeros((5, 6, 2))}
    data["PE x"][1,1,0] = 1
    data["PE y"][1,1,0] = 1

    # print("step1", data["PE x"][:,:,0])
    print("step1", data["PE y"][:,:,0])
    data = trafo(data)
    print("step2", data["PE x"])
    print("step2", data["PE y"])