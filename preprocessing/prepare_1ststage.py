import argparse
import logging
import os
import pathlib
import time

import h5py
import numpy as np
import torch
import yaml
from typing import Union
from tqdm.auto import tqdm

from data_stuff.transforms import (ComposeTransform, NormalizeTransform,
                             PowerOfTwoTransform, ReduceTo2DTransform, CutLengthTransform,
                             SignedDistanceTransform, PositionalEncodingTransform, ToTensorTransform)
from data_stuff.utils import SettingsTraining
from preprocessing.prepare_paths import Paths1HP, Paths2HP

def prepare_dataset_for_1st_stage(paths: Paths1HP, settings: SettingsTraining, info_file: str = "info.yaml"):
    time_begin = time.perf_counter()
    info_file_path = settings.model / info_file

    if settings.problem == "extend1":
        cutlengthtrafo=True
    else:
        cutlengthtrafo=False

    if settings.case == "test" or settings.case_2hp:
        # get info of training
        with open(info_file_path, "r") as file:
            info = yaml.safe_load(file)
    else:
        info = None
            
    # TODO unsauber, TODO cutlengthtrafo zu länge die in info.yaml gespeichert ist
    prepare_dataset(paths, settings.inputs, power2trafo=False, cutlengthtrafo=cutlengthtrafo, box_length=settings.len_box,info=info)
    
    if settings.case == "train" and not settings.case_2hp:
        # store info of training
        with open(settings.destination / info_file, "w") as file:
            yaml.safe_dump(info, file)

    time_end = time.perf_counter() - time_begin
    with open(paths.dataset_1st_prep_path / "preparation_time.yaml", "w") as file:
        yaml.safe_dump(
            {"timestamp of end": time.ctime(), 
                "duration of whole process in seconds": time_end}, file)
        

def prepare_dataset(paths: Union[Paths1HP, Paths2HP], inputs: str, power2trafo: bool = True, cutlengthtrafo: bool = False, box_length: int = 256, info:dict = None):
    """
    Create a dataset from the raw pflotran data in raw_data_path.
    The saved dataset is normalized using the mean and standard deviation, which are saved to info.yaml in the new dataset folder.

    Parameters
    ----------
        raw_data_path : str
            Path to the raw pflotran data directory.
        datasets_path : str
            Path to the directory where all dataset are saved.
        dataset_name : str
            Name of the raw data. This will also be the name of the new dataset.
        input_variables : str
            String of characters, each of which is either x, y, z, p, t, k, i, s, g.
            TODO make g (pressure gradient cell-size independent?)
    """
    time_start = time.perf_counter()
    check_for_dataset(paths.raw_path)
    dataset_prepared_path = pathlib.Path(paths.dataset_1st_prep_path)
    dataset_prepared_path.mkdir(parents=True, exist_ok=True)
    dataset_prepared_path.joinpath("Inputs").mkdir(parents=True, exist_ok=True) # TODO
    dataset_prepared_path.joinpath("Labels").mkdir(parents=True, exist_ok=True)

    transforms = get_transforms(reduce_to_2D=True, reduce_to_2D_xy=True, power2trafo=power2trafo, cutlengthtrafo=cutlengthtrafo, box_length=box_length)
    inputs = expand_property_names(inputs)
    time_first = "   0 Time  0.00000E+00 y"
    time_final = "   3 Time  5.00000E+00 y"
    time_steady_state = "   4 Time  2.75000E+01 y"
    pflotran_settings = get_pflotran_settings(paths.raw_path)
    dims = np.array(pflotran_settings["grid"]["ncells"])
    total_size = np.array(pflotran_settings["grid"]["size"])
    cell_size = total_size/dims

    if info is None: calc = WelfordStatistics()
    tensor_transform = ToTensorTransform()
    output_variables = ["Temperature [C]"]
    data_paths, runs = detect_datapoints(paths.raw_path)
    total = len(data_paths)
    for data_path, run in tqdm(zip(data_paths, runs), desc="Converting", total=total):
        x = load_data(data_path, time_first, inputs, dims)
        y = load_data(data_path, time_steady_state, output_variables, dims)
        loc_hp = get_hp_location(x)
        x = transforms(x, loc_hp=loc_hp)
        if info is None: calc.add_data(x) 
        x = tensor_transform(x)
        y = transforms(y, loc_hp=loc_hp)
        if info is None: calc.add_data(y)
        y = tensor_transform(y)
        torch.save(x, os.path.join(dataset_prepared_path, "Inputs", f"{run}.pt"))
        torch.save(y, os.path.join(dataset_prepared_path, "Labels", f"{run}.pt"))
        
    if info is not None: 
        info["CellsNumberPrior"] = info["CellsNumber"]
        info["PositionHPPrior"] = info["PositionLastHP"]
        assert info["CellsSize"][:2] == cell_size.tolist()[:2], f"Cell size changed between given info.yaml {info['CellsSize']} and data {cell_size.tolist()}"
    else:
        info = dict()
        means = calc.mean()
        stds = calc.std()
        mins = calc.min()
        maxs = calc.max()
        info["Inputs"] = {key: {"mean": means[key],
                                "std": stds[key],
                                "min": mins[key],
                                "max": maxs[key],
                                "norm": get_normalization_type(key),
                                "index": n}
                        for n, key in enumerate(inputs)}
        info["Labels"] = {key: {"mean": means[key],
                                "std": stds[key],
                                "min": mins[key],
                                "max": maxs[key],
                                "norm": get_normalization_type(key),
                                "index": n}
                        for n, key in enumerate(output_variables)}
        
    info["CellsSize"] = cell_size.tolist()
    # change of size possible; order of tensor is in any case the other way around
    assert 1 in y.shape, "y is not expected to have several output parameters"
    assert len(y.shape) == 3, "y is expected to be 2D"
    dims = list(y.shape)[1:]
    info["CellsNumber"] = dims
    try:
        info["PositionLastHP"] = loc_hp.tolist()
    except:
        info["PositionLastHP"] = loc_hp
    # info["PositionLastHP"] = get_hp_location_from_tensor(x, info)
    with open(dataset_prepared_path / "info.yaml", "w") as file:
        yaml.dump(info, file)

    normalize(dataset_prepared_path, info, total)
    
    time_end = time.perf_counter()
    with open(dataset_prepared_path / "args.yaml", "w") as f:
        yaml.dump({"dataset":paths.raw_path.name, "inputs": inputs}, f, default_flow_style=False)
        f.write(f"Duration for preparation in sec: {time_end-time_start}")

    return info

## helper functions
def check_for_dataset(path: pathlib.Path) -> str:
    """
    Check if the dataset exists and is not empty.
    Dataset should be in the following folder self.dataset_path/<dataset_name>
    """
    if not path.exists():
        raise ValueError(
            f"Dataset {path.name} does not exist in {path}")
    if not any(path.iterdir()):
        raise ValueError(f"Dataset {path.name} is empty")

def detect_datapoints(dataset_path_raw: pathlib.Path):
    """
    Create the simulation dataset by preparing a list of samples
    Simulation data are sorted in an ascending order by run number
    :returns: (data_paths, runs) where:
        - data_paths is a list containing paths to all simulation runs in the dataset, NOT the actual simulated data
        - runs is a list containing one label per run
    """
    set_data_paths_runs, runs = [], []
    found_dataset = False

    for folder in dataset_path_raw.iterdir():
        if folder.is_dir():
            for file in folder.iterdir():
                if file.name == "pflotran.h5":
                    set_data_paths_runs.append((folder.name, file))
                    found_dataset = True
    # Sort the data and runs in ascending order
    set_data_paths_runs = sorted(
        set_data_paths_runs, key=lambda val: int(val[0].strip('RUN_')))
    runs = [data_path[0] for data_path in set_data_paths_runs]
    data_paths = [data_path[1] for data_path in set_data_paths_runs]
    if not found_dataset:
        raise ValueError(f"No dataset found in {dataset_path_raw}")

    return data_paths, runs

def expand_property_names(properties: str):
    translation = {
        "x": "Liquid X-Velocity [m_per_y]",
        "y": "Liquid Y-Velocity [m_per_y]",
        "z": "Liquid Z-Velocity [m_per_y]",
        "p": "Liquid Pressure [Pa]",
        "t": "Temperature [C]",
        "k": "Permeability X [m^2]",
        "i": "Material ID",
        "s": "SDF",
        "a": "PE x", # positional encoding: signed distance in x direction
        "b": "PE y", # positional encoding: signed distance in y direction
        "g": "Pressure Gradient [-]",
        "o": "Original Temperature [C]"
    }
    possible_vars = ','.join(translation.keys())
    assert all((prop in possible_vars)
               for prop in properties), f"input parameters have to be a string of characters, each of which is either {possible_vars}"
    return [translation[prop] for prop in properties]

def get_normalization_type(property:str):
    """
    Returns the normalization type for a given property
    Types can be:
        Rescale: Rescale the data to be between 0 and 1
        Standardize: Standardize the data to have mean 0 and standard deviation 1
        None: Do not normalize the data
    """
    types = {
        "default": "Rescale", #Standardize
        # "Material ID": "Rescale",
        "SDF": None,
        "PE x": None,
        "PE y": None,
        "Original Temperature [C]": None,
    }
    
    if property in types:
        return types[property]
    else:
        return types["default"]

def get_pflotran_settings(dataset_path_raw: str):
    with open(dataset_path_raw / "inputs" / "settings.yaml", "r") as f:
        pflotran_settings = yaml.safe_load(f)
    return pflotran_settings

def load_data(data_path: str, time: str, variables: dict, dimensions_of_datapoint: tuple):
    """
    Load data from h5 file on data_path, but only the variables named in variables.get_ids() at time stamp variables.time
    Sets the values of each PhysicalVariable in variables to the loaded data.
    """
    data = dict()
    with h5py.File(data_path, "r") as file:
        for key in variables:  # properties
            try:
                data[key] = torch.tensor(np.array(file[time][key]).reshape(
                    dimensions_of_datapoint, order='F')).float()
            except KeyError:
                if key == "SDF":
                    data[key] = torch.tensor(np.array(file[time]["Material ID"]).reshape(dimensions_of_datapoint, order='F')).float()
                elif key in ["PE x", "PE y"]:
                    data[key] = torch.tensor(np.array(file[time]["Material ID"]).reshape(dimensions_of_datapoint, order='F')).float()
                elif key == "Pressure Gradient [-]":
                    empty_field = torch.ones(list(dimensions_of_datapoint)).float()
                    pressure_grad = get_pressure_gradient(data_path)
                    data[key] = empty_field * pressure_grad[1]
                elif key == "Original Temperature [C]":
                    empty_field = torch.ones(list(dimensions_of_datapoint)).float()
                    data[key] = empty_field * 0 #10.6
                    # TODO use torch.zeros instead
                else:
                    raise KeyError(
                        f"Key '{key}' not found in {data_path} at time {time}")
    return data

def get_hp_location(data):
    try:  # TODO problematic with SDF?
        ids = data["Material ID"]
    except:
        try:
            ids = data["SDF"]
        except:
            return None
    max_id = ids.max()
    loc_hp = np.array(np.where(ids == max_id)).squeeze()
    return loc_hp

def get_hp_location_from_tensor(data: torch.Tensor, info: dict):
    try:
        idx = info["Inputs"]["Material ID"]["index"]
    except:
        idx = info["Inputs"]["SDF"]["index"]
    loc_hp = torch.Tensor(torch.where(data[idx] == data[idx].max())).squeeze().int().tolist()
    return loc_hp

def get_pressure_gradient(data_path):
    pressure_grad_file = data_path.parent / "pressure_gradient.txt"
    pressure_grad_file_interim = data_path.parent / "interim_pressure_gradient.txt"
    try:
        with open(pressure_grad_file, "r") as f:
            pressure_grad = f.read().split()[1:]
    except FileNotFoundError:
        with open(pressure_grad_file_interim, "r") as f:
            pressure_grad = f.read().split()[1:]
    pressure_grad = torch.tensor([float(grad) for grad in pressure_grad])

    return pressure_grad


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
            result[key] = (np.sqrt(self.var()[key])).item()
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

def get_transforms(reduce_to_2D: bool, reduce_to_2D_xy: bool, power2trafo: bool = True, cutlengthtrafo: bool = False, box_length:int=256):
    transforms_list = []

    if reduce_to_2D:
        transforms_list.append(ReduceTo2DTransform(
            reduce_to_2D_xy=reduce_to_2D_xy))
    if power2trafo:
        transforms_list.append(PowerOfTwoTransform())
    if cutlengthtrafo:
        transforms_list.append(CutLengthTransform(box_length))
    transforms_list.append(SignedDistanceTransform())
    transforms_list.append(PositionalEncodingTransform())

    transforms = ComposeTransform(transforms_list)
    return transforms

def normalize(dataset_path: str, info: dict, total: int = None):
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
    dataset_path = pathlib.Path(dataset_path)
    input_path = dataset_path.joinpath("Inputs")
    label_path = dataset_path.joinpath("Labels")
    for input_file in tqdm(input_path.iterdir(), desc="Normalizing inputs", total=total):
        x = torch.load(input_file)
        x = norm(x,"Inputs")
        torch.save(x, input_file)
    for label_file in tqdm(label_path.iterdir(), desc="Normalizing labels", total=total):
        y = torch.load(label_file)
        y = norm(y,"Labels")
        torch.save(y, label_file)


# if __name__ == "__main__":
#     if os.path.exists("/scratch/sgs/pelzerja/"):
#         default_raw_dir = "/scratch/sgs/pelzerja/datasets/1hp_boxes"
#         default_target_dir="/home/pelzerja/pelzerja/test_nn/datasets_prepared/1HP_NN/experiments"
#     else:
#         default_raw_dir = "/home/pelzerja/Development/simulation_groundtruth_pflotran/Phd_simulation_groundtruth/datasets/1hp_boxes"
#         default_target_dir = "/home/pelzerja/Development/datasets_prepared/1HP_NN"
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--raw_dir", type=str, default=default_raw_dir)
#     parser.add_argument("--datasets_dir", type=str, default=default_target_dir)
#     parser.add_argument("--dataset_name", type=str, default="benchmark_dataset_2d_10datapoints")
#     parser.add_argument("--inputs_prep", type=str, default="pksi")
#     args = parser.parse_args()
#     args = SettingsPrepare(**vars(args))
#     prepare_dataset(args)