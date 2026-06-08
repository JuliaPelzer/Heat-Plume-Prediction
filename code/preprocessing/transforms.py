"""
Definition of problem-specific transform classes
"""

from code.utils import logging as log  # noqa: F401
from pathlib import Path

import torch
from torch import nonzero
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


class SignedDistanceTransform:
    """
    Transform class to calculate signed distance transform for material id.
    This transform takes a dict of tensors as input and returns a dict of tensors.
    """

    def __init__(self):
        pass

    def __call__(self, data: dict):
        log.info("Start SignedDistanceTransform")

        # check if Material ID is in data (inputs vs. labels)
        if "Material ID" not in data.keys():
            log.info("No material ID in data, no SignedDistanceTransform")
            return data

        def get_loc_hp():  # TODO clean up - this appearsquite often in various forms
            if "Material ID" in data.keys():
                loc_hp = nonzero(data["Material ID"] == torch.max(data["Material ID"]))
                if loc_hp.numel() == 0:
                    return None
                if loc_hp.shape[0] > 1:
                    log.info(f"loc_hp returns more than one position: {loc_hp.shape[0]}")
                log.info(f"loc_hp: {loc_hp}")
                return loc_hp

        loc_hp = get_loc_hp()
        if loc_hp is None:
            log.info("No hp location found, skipping SignedDistanceTransform")
            return data
        log.info(f"Keys in data: {data.keys()}")
        data["SDF"] = self.sdf(data["SDF"].float(), loc_hp.float())
        log.info("SignedDistanceTransform done")
        return data

    def sdf(self, data: torch.tensor, loc_hp: torch.tensor):
        # loc_hp: (N, D), data: (H, W) or (D1, D2, D3)
        dims = data.dim()
        if dims not in (2, 3):
            raise ValueError(f"SDF expects 2D or 3D data, got {dims}D")

        if loc_hp.dim() == 1:
            loc_hp = loc_hp.unsqueeze(0)

        # build grid of coordinates
        coords = [torch.arange(s, device=data.device, dtype=data.dtype) for s in data.shape]
        grid = torch.stack(torch.meshgrid(*coords, indexing="ij"), dim=-1)  # (*shape, D)
        grid_flat = grid.reshape(-1, dims)

        # compute distance to closest hp for each grid point
        dists = torch.cdist(grid_flat.unsqueeze(0), loc_hp.unsqueeze(0)).squeeze(0)  # (P, N)
        min_dist = dists.min(dim=1).values.reshape(data.shape)

        min_dist = 1 - min_dist / min_dist.max()
        return min_dist


def get_transforms(reduce_to_2D: bool = True, inputs=None):
    transforms_list = []
    if reduce_to_2D:
        transforms_list.append(ReduceTo2DTransform())
    log.info(f"Inputs for get_transforms: {inputs}")
    if inputs:
        if isinstance(inputs, str):
            has_sdf = "s" in inputs
        else:
            has_sdf = "sdf" in inputs
        if has_sdf:
            transforms_list.append(SignedDistanceTransform())

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
