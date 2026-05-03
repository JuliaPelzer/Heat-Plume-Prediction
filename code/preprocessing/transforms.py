"""
Definition of problem-specific transform classes
"""

from code.utils import logging as log  # noqa: F401
from pathlib import Path

import torch
from tqdm import tqdm


class NormalizeTransform:
    def __init__(self, info: dict, out_range: tuple[float, float] = (0, 1)):
        self.info = info
        self.out_min, self.out_max = out_range

    def __call__(self, data, data_type="Inputs"):
        for _prop, stats in self.info[data_type].items():
            index = stats["index"]
            if index < data.shape[0]:
                self.__apply_norm(data, index, stats)
            else:
                log.warning(f"Index {index} might be in training data but not in this dataset")
        return data

    def reverse(self, data, data_type="Labels"):
        for _prop, stats in self.info[data_type].items():
            index = stats["index"]
            self.__reverse_norm(data, index, stats)
        return data

    def __apply_norm(self, data, index, stats):
        norm = stats["norm"]

        def rescale():
            delta = stats["max"] - stats["min"]
            if 0 == delta:
                raise ValueError("Cannot rescale data with zero range (max equals min).")
            data[index] = (data[index] - stats["min"]) / delta * (self.out_max - self.out_min) + self.out_min

        if norm == "LogRescale":
            data[index] = torch.log(data[index] - stats["min"] + 1)
            rescale()
        elif norm == "Rescale":
            rescale()
        elif norm == "Standardize":
            data[index] = (data[index] - stats["mean"]) / stats["std"]
        elif norm is None:
            pass
        else:
            raise ValueError(f"Normalization type '{stats['norm']}' not recognized")

    def __reverse_norm(self, data, index, stats):
        norm = stats["norm"]

        def rescale():
            delta = stats["max"] - stats["min"]
            data[index] = (data[index] - self.out_min) / (self.out_max - self.out_min) * delta + stats["min"]

        if norm == "LogRescale":
            rescale()
            data[index] = torch.exp(data[index]) + stats["min"] - 1
        elif norm == "Rescale":
            rescale()
        elif norm == "Standardize":
            data[index] = data[index] * stats["std"] + stats["mean"]
        elif norm is None:
            pass
        else:
            raise ValueError(f"Normalization type '{stats['Norm']}' not recognized")


class ReduceTo2DTransform:
    """
    Transform class to reduce data to 2D, reduce in x, in height of hp: x=7
    This Transform takes a dict of tensors as input and returns a dict of tensors
    """

    def __init__(self):
        # if reduce_to_2D_wrong then the data will still be reduced to 2D but in x,y dimension instead of y,z
        self.slice_dimension = 2

    def __call__(self, data, loc_hp: tuple):
        log.info("Start ReduceTo2DTransform")
        already_2d: bool = False

        for data_prop in data.keys():
            # check if data is already 2D, if so: do nothing/ only switch axes (for plotting)
            data_shape = data[data_prop].shape
            if 1 in data_shape or len(data_shape) == 2:
                already_2d = True

        if not already_2d:
            if loc_hp is not None:
                self.loc_hp_slice = loc_hp[self.slice_dimension]

            for prop in data.keys():
                data[prop].transpose_(0, 2)
            for prop in data.keys():
                assert self.loc_hp_slice <= data[prop].shape[0], (
                    "ReduceTo2DTransform: x is larger than data dimension 0"
                )
                data[prop] = data[prop][self.loc_hp_slice, :, :]
                data[prop] = torch.unsqueeze(data[prop], 0)
        log.info("Reduced data to 2D, but still has dummy dimension 0 for Normalization to work")
        return data


class ComposeTransform:
    """Transform class that combines multiple other transforms into one"""

    def __init__(self, transforms):
        """
        :param transforms: transforms to be combined
        """
        self.transforms = transforms

    def __call__(self, data, loc_hp: tuple = None):
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
            except AttributeError:
                pass
        return data


class ToTensorTransform:
    """Transform class to convert dict of tensors to one tensor"""

    def __init__(self):
        pass

    def __call__(self, data: dict):
        log.info("Start ToTensorTransform")
        result: torch.Tensor = None
        for prop in data.keys():
            if result is None:
                result = data[prop].squeeze()[None, ...]
            else:
                result = torch.cat((result, data[prop].squeeze()[None, ...]), axis=0)
        log.info("Converted data to torch.Tensor")
        return result


def get_transforms(reduce_to_2D: bool = True):
    transforms_list = []
    if reduce_to_2D:
        transforms_list.append(ReduceTo2DTransform())

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
        x = norm(x, "Inputs")
        torch.save(x, input_file)
    for label_file in tqdm((dataset_path / "Labels").iterdir(), desc="Normalizing labels", total=total):
        y = torch.load(label_file)
        y = norm(y, "Labels")
        torch.save(y, label_file)
