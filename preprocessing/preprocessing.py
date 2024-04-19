from argparse import Namespace
import numpy as np
import torch
from pathlib import Path
import yaml
from tqdm.auto import tqdm

from preprocessing.transforms import (get_transforms, normalize, ToTensorTransform)
from utils.utils_args import is_empty, load_yaml, save_yaml
from preprocessing.statistics import WelfordStatistics
import preprocessing.load_data as load
import utils.utils_args as ut
from preprocessing.prepare_allin1_copy import preprocessing_allin1

def preprocessing(args):
    print("Preparing dataset")
        
    if args.problem in ["test", "1hp", "extend"]:
        if is_unprepared(args.data_prep): # or args.case == "test":
            info = load_yaml(args.model / "info.yaml") if args.case != "train" else None
            info = prepare_dataset(args, info=info)
        else:
            info = load_yaml(args.data_prep / "info.yaml")

    elif args.problem == "2stages":
        exit("2stages not implemented yet, use other branch")

    elif args.problem == "allin1":
        # only doable if 3 datasets exist (case: different datasets)
        # todo handling of different datasets and several stages
        info = prepare_allin1(args)

    if args.case == "train": # TODO also finetune?
        ut.save_yaml(args.model / "info.yaml", info)

    print("Dataset prepared")


def prepare_allin1(args):
    # achtung mit is_prepared_checks
    # TODO LOGIK CHECKEN
    if args.case != "infer":
        info = prepare_allin1_block(args, args.data_raw + "_train" + " inputs_" + args.inputs)# correct???
        if args.case == "train": # TODO also finetune?
            ut.save_yaml(args.model / "info.yaml", info)

        prepare_allin1_block(args, args.data_raw + "_val" + " inputs_" + args.inputs)
        prepare_allin1_block(args, args.data_raw + "_test" + " inputs_" + args.inputs)


def prepare_allin1_block(args, data_prep:Path):
    if is_unprepared(data_prep):
        if "n" in args.inputs:
            additional_input_unnormed = preprocessing_allin1(args, data_prep) # TODO NEXT
        else:
            additional_input_unnormed = None
        info = load_yaml(args.model / "info.yaml") if args.case != "train" else None
        info = prepare_dataset(args, info=info, additional_input=additional_input_unnormed)
    else:
        info = load_yaml(data_prep / "info.yaml")
    
    return info

# helper function
def is_unprepared(path:Path):
    return is_empty(path / "Inputs") or is_empty(path / "Labels") or not (path / "info.yaml").exists()

def prepare_dataset(args, info:dict = None, additional_input: torch.Tensor = None):
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
            String of characters, each of which is either x, y, z, p, t, k, i, s, g, ...
            TODO make g (pressure gradient cell-size independent?)
    """
    transforms = get_transforms(problem=args.problem)
    inputs = expand_property_names(args.inputs)
    time_init = "   0 Time  0.00000E+00 y"
    time_prediction = "   4 Time  2.75000E+01 y" #  "   3 Time  5.00000E+00 y"  # 
    pflotran_settings = load_yaml(args.data_raw / "inputs" / "settings.yaml")
    dims = np.array(pflotran_settings["grid"]["ncells"])
    total_size = np.array(pflotran_settings["grid"]["size"])
    cell_size = total_size/dims

    if info is None: calc = WelfordStatistics()
    tensor_transform = ToTensorTransform()
    output_variables = ["Temperature [C]"]
    data_paths, runs = load.detect_datapoints(args.data_raw)
    total = len(data_paths)
    for data_path, run in tqdm(zip(data_paths, runs), desc="Converting", total=total):
        x = load.load_data(data_path, time_init, inputs, dims, additional_input=additional_input)
        y = load.load_data(data_path, time_prediction, output_variables, dims)
        loc_hp = load.get_hp_location(x)
        x = transforms(x, loc_hp=loc_hp)
        if info is None: calc.add_data(x) 
        x = tensor_transform(x)
        y = transforms(y, loc_hp=loc_hp)
        if info is None: calc.add_data(y)
        y = tensor_transform(y)
        torch.save(x, args.data_prep / "Inputs" / f"{run}.pt")
        torch.save(y, args.data_prep / "Labels" / f"{run}.pt")
        
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
    save_yaml(args.data_prep / "info.yaml", info)
    normalize(args.data_prep, info, total)
    save_yaml(args.data_prep / "args.yaml", {"dataset":args.data_raw.name, "inputs": inputs})

    return info

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
        "o": "Original Temperature [C]",
        "m": "MDF",
        "l": "LST",
        "n": "Preprocessed Temperature [C]",
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
        "MDF": None,
        "LST": None,
        "Original Temperature [C]": None,
    }
    
    if property in types:
        return types[property]
    else:
        return types["default"]