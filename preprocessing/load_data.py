import h5py
import numpy as np
import torch
from pathlib import Path

def load_data(data_path: Path, time: str, variables: dict, dimensions_of_datapoint: tuple, additional_input: torch.Tensor = None, print_bool: bool = False, device:str = "cpu"):
    """
    Load data from h5 file on data_path, but only the variables named in variables.get_ids() at time stamp variables.time
    Sets the values of each PhysicalVariable in variables to the loaded data.
    """
    data = dict()
    with h5py.File(data_path, "r") as file:
        for key in variables:  # properties
            try:
                data[key] = torch.tensor(np.array(file[time][key]).reshape(dimensions_of_datapoint, order='F')).float()
            except KeyError:
                if key == "SDF":
                    data[key] = torch.tensor(np.array(file[time]["Material ID"]).reshape(dimensions_of_datapoint, order='F')).float()
                elif key in ["PE x", "PE y", "MDF", "LST"]:
                    data[key] = torch.tensor(np.array(file[time]["Material ID"]).reshape(dimensions_of_datapoint, order='F')).float()
                elif key == "Pressure Gradient [-]":
                    empty_field = torch.ones(list(dimensions_of_datapoint)).float()
                    pressure_grad = get_pressure_gradient(data_path)
                    data[key] = empty_field * pressure_grad[1]
                elif key == "Original Temperature [C]":
                    empty_field = torch.ones(list(dimensions_of_datapoint)).float()
                    data[key] = empty_field * 0 #10.6
                    # TODO use torch.zeros instead
                elif key == "Preprocessed Temperature [C]":
                    data[key] = additional_input.float()
                else:
                    raise KeyError(
                        f"Key '{key}' not found in {data_path} at time {time}")
            if print_bool:
                print(f"Loaded {key} at time {time} with shape {data[key].shape}")

    return data

def get_hp_location(data):
    try:
        ids = data["Material ID"]
    except:
        try:
            ids = data["SDF"]
        except:
            try:
                ids = data["MDF"]
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


def detect_datapoints(dataset_path_raw: Path):
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