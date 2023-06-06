import os
import logging
import h5py
import numpy as np
import yaml
import torch
from data.transforms import NormalizeTransform, ComposeTransform, ReduceTo2DTransform, ToTensorTransform, SignedDistanceTransform, PowerOfTwoTransform
from tqdm.auto import tqdm
import pathlib
import argparse


def prepare_dataset(raw_data_directory: str, datasets_path: str, dataset_name: str, input_variables: str, name_extension: str = "", power2trafo: bool = True, info:dict = None):
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
    full_raw_path = check_for_dataset(raw_data_directory, dataset_name)
    datasets_path = pathlib.Path(datasets_path)
    new_dataset_path = datasets_path.joinpath(dataset_name+name_extension)
    new_dataset_path.mkdir(parents=True, exist_ok=True)
    new_dataset_path.joinpath("Inputs").mkdir(parents=True, exist_ok=True)
    new_dataset_path.joinpath("Labels").mkdir(parents=True, exist_ok=True)

    transforms = get_transforms(reduce_to_2D=True, reduce_to_2D_xy=True, power2trafo=power2trafo)
    input_variables = expand_property_names(input_variables)
    time_first = "   0 Time  0.00000E+00 y"
    time_final = "   3 Time  5.00000E+00 y"
    time_steady_state = "   4 Time  2.75000E+01 y"
    pflotran_settings = get_pflotran_settings(full_raw_path)
    dims = np.array(pflotran_settings["grid"]["ncells"])
    total_size = np.array(pflotran_settings["grid"]["size"])
    cell_size = total_size/dims

    if info is None: calc = WelfordStatistics()
    tensor_transform = ToTensorTransform()
    output_variables = ["Temperature [C]"]
    datapaths, runs = detect_datapoints(full_raw_path)
    total = len(datapaths)
    for datapath, run in tqdm(zip(datapaths, runs), desc="Converting", total=total):
        x = load_data(datapath, time_first, input_variables, dims)
        y = load_data(datapath, time_steady_state, output_variables, dims)
        loc_hp = get_hp_location(x)
        x = transforms(x, loc_hp=loc_hp)
        if info is None: calc.add_data(x) 
        x = tensor_transform(x)
        y = transforms(y, loc_hp=loc_hp)
        if info is None: calc.add_data(y)
        y = tensor_transform(y)
        torch.save(x, os.path.join(new_dataset_path, "Inputs", f"{run}.pt"))
        torch.save(y, os.path.join(new_dataset_path, "Labels", f"{run}.pt"))
        
    if info is not None: 
        info["CellsNumberPrior"] = info["CellsNumber"]
        info["PositionHPPrior"] = info["PositionLastHP"]
        assert info["CellsSize"] == cell_size.tolist(), "Cell size changed between given info.yaml and data"
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
                        for n, key in enumerate(input_variables)}
        info["Labels"] = {key: {"mean": means[key],
                                "std": stds[key],
                                "min": mins[key],
                                "max": maxs[key],
                                "norm": get_normalization_type(key),
                                "index": n}
                        for n, key in enumerate(output_variables)}
        
    info["CellsSize"] = cell_size.tolist()
    if power2trafo: 
        # change of size possible; order of tensor is in any case the other way around
        assert 1 in y.shape, "y is not expected to have several output parameters"
        assert len(y.shape) == 3, "y is expected to be 2D"
        dims = [y.shape[2], y.shape[1], y.shape[0]]
    else:
        dims = dims.tolist()
    info["CellsNumber"] = dims
    info["PositionLastHP"] = loc_hp.tolist()
    with open(os.path.join(new_dataset_path, "info.yaml"), "w") as file:
        yaml.dump(info, file)
    normalize(new_dataset_path, info, total)

## helper functions
def check_for_dataset(path: str, name: str) -> str:
    """
    Check if the dataset exists and is not empty.
    Dataset should be in the following folder self.dataset_path/<dataset_name>
    """
    dataset_path_full = os.path.join(path, name)
    if not os.path.exists(dataset_path_full):
        raise ValueError(
            f"Dataset {name} does not exist in {dataset_path_full}")
    if len(os.listdir(dataset_path_full)) == 0:
        raise ValueError(f"Dataset {name} is empty")
    return dataset_path_full

def detect_datapoints(raw_dataset_path: str):
    """
    Create the simulation dataset by preparing a list of samples
    Simulation data are sorted in an ascending order by run number
    :returns: (data_paths, runs) where:
        - data_paths is a list containing paths to all simulation runs in the dataset, NOT the actual simulated data
        - runs is a list containing one label per run
    """
    set_data_paths_runs, runs = [], []
    found_dataset = False
    raw_dataset_path = pathlib.Path(raw_dataset_path)

    logging.info(f"Directory of currently used dataset is: {raw_dataset_path}")
    for _, folders, _ in os.walk(raw_dataset_path):
        for folder in folders:
            for file in os.listdir(raw_dataset_path.joinpath(folder)):
                if file == "pflotran.h5":
                    set_data_paths_runs.append(
                        (folder, raw_dataset_path.joinpath(folder, file)))
                    found_dataset = True
    # Sort the data and runs in ascending order
    set_data_paths_runs = sorted(
        set_data_paths_runs, key=lambda val: int(val[0].strip('RUN_')))
    runs = [data_path[0] for data_path in set_data_paths_runs]
    data_paths = [data_path[1] for data_path in set_data_paths_runs]
    if not found_dataset:
        raise ValueError(f"No dataset found in {raw_dataset_path}")
    assert len(data_paths) == len(runs)

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
        "g": "Pressure Gradient [-]"
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
    }
    
    if property in types:
        return types[property]
    else:
        return types["default"]

def get_pflotran_settings(raw_dataset_path: str):
    raw_dataset_path = pathlib.Path(raw_dataset_path)
    with open(raw_dataset_path.joinpath("inputs", "settings.yaml"), "r") as f:
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
                    data[key] = torch.tensor(np.array(file[time]["Material ID"]).reshape(
                        dimensions_of_datapoint, order='F')).float()
                elif key == "Pressure Gradient [-]":
                    empty_field = torch.ones(list(dimensions_of_datapoint)).float()
                    pressure_grad = get_pressure_gradient(data_path)
                    data[key] = empty_field * pressure_grad[1]
                else:
                    raise KeyError(
                        f"Key '{key}' not found in {data_path} at time {time}")
    return data

def get_hp_location(data):
    try:  # TODO problematic with SDF?
        ids = data["Material ID"]
    except:
        ids = data["SDF"]
    max_id = ids.max()
    loc_hp = np.array(np.where(ids == max_id)).squeeze()
    return loc_hp

def get_pressure_gradient(data_path):
    pressure_grad_file = data_path.parent.joinpath("pressure_gradient.txt")
    with open(pressure_grad_file, "r") as f:
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

def get_transforms(reduce_to_2D: bool, reduce_to_2D_xy: bool, power2trafo: bool = True):
    transforms_list = []

    if reduce_to_2D:
        transforms_list.append(ReduceTo2DTransform(
            reduce_to_2D_xy=reduce_to_2D_xy))
    if power2trafo:
        transforms_list.append(PowerOfTwoTransform())
    transforms_list.append(SignedDistanceTransform())

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


if __name__ == "__main__":

    # TODO reasonable defaults
    remote = False
    if os.path.exists("/scratch/sgs/pelzerja/"):
        remote = True
    default_raw_dir = "/home/pelzerja/Development/simulation_groundtruth_pflotran/Phd_simulation_groundtruth/datasets/1hp_boxes"
    default_target_dir = "/home/pelzerja/Development/datasets_prepared/1HP_NN"
    if remote:
        default_raw_dir = "/scratch/sgs/pelzerja/datasets/1hp_boxes"
        default_target_dir="/home/pelzerja/pelzerja/test_nn/datasets_prepared/1HP_NN"
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default=default_raw_dir)
    parser.add_argument("--datasets_dir", type=str, default=default_target_dir)
    parser.add_argument("--dataset_name", type=str, default="benchmark_dataset_2d_100dp_vary_perm")
    parser.add_argument("--inputs", type=str, default="pksi")
    args = parser.parse_args()
    prepare_dataset(
        args.raw_dir,
        args.datasets_dir,
        args.dataset_name,
        args.inputs,
        )
