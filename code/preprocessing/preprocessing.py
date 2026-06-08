from code.preprocessing.preparing_datasets import raw_data_loading as load
from code.preprocessing.preparing_datasets.statistics import WelfordStatistics
from code.preprocessing.transforms import ToTensorTransform, get_transforms, normalize
from code.utils import logging as log  # noqa: F401
from code.utils.utils_args import is_empty, load_time_steps, load_time_steps_full_str, load_yaml, save_yaml
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm


def preprocessing(args: dict):
    log.info("Preparing dataset")
    network = args["network"]
    if is_unprepared(args["data_prep"]):
        info = load_yaml(args["model"] / "info.yaml") if args["case"] != "train" else None

        if network in ["convlstm", "rnn", "lstm"]:
            info = prepare_dataset_for_sequence(args, info=info)
            log.info("Preparing dataset...")
        else:
            info = prepare_dataset(args, info=info)
    else:
        info = load_yaml(args["data_prep"] / "info.yaml")
    log.info(f"Dataset prepared: {args['data_prep']}")

    if args["case"] == "train":
        save_yaml(info, args["destination"] / "info.yaml")
    return info


# helper function
def is_unprepared(path: Path):
    (path / "Inputs").mkdir(parents=True, exist_ok=True)
    (path / "Labels").mkdir(parents=True, exist_ok=True)
    return is_empty(path / "Inputs") or is_empty(path / "Labels") or not (path / "info.yaml").exists()


def get_time_prediction(data_path):
    with h5py.File(data_path, "r") as file:
        for item in file.keys():
            if "2.75000E+01" in item or "2.50000E+01" in item:
                return item
    raise ValueError("Could not find time prediction in h5 file")


def prepare_dataset(args: dict, info: dict = None):
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
    """

    transforms = get_transforms(inputs=args["inputs"])
    inputs = expand_property_names(args["inputs"])
    outputs = expand_property_names(args["outputs"])
    time_init = "   0 Time  0.00000E+00 y"

    try:  # old data (before 2025)
        pflotran_settings = load_yaml(args["data_raw"] / "inputs" / "settings.yaml")
        total_size = np.array(pflotran_settings["grid"]["size"])
        dims = np.array(pflotran_settings["grid"]["ncells"])
        cell_size = total_size / dims
        resolution = cell_size[0]
    except Exception: # new data (2025)
        pflotran_settings = load_yaml(args["data_raw"] / "settings.yaml")
        resolution = 5  # [m] # goal-resolution
        total_size = np.array([*pflotran_settings["grid"]["size [m]"], resolution])
        cell_size = resolution * np.ones(len(total_size))
        dims = (total_size / cell_size).astype(int)

    if info is None:
        calc = WelfordStatistics()
    tensor_transform = ToTensorTransform()
    data_paths, runs = load.detect_datapoints(args["data_raw"])
    total = len(data_paths)
    print_bool = True
    for data_path, run in tqdm(zip(data_paths, runs, strict=True), desc="Converting", total=total):
        time_prediction = get_time_prediction(data_path)
        x = load.load_raw_data(data_path, time_init, inputs, dims, time_prediction, print_bool=print_bool)
        y = load.load_raw_data(data_path, time_prediction, outputs, dims, time_prediction, print_bool=print_bool)

        # For step4: if temperature is requested in inputs, replace with step3 prediction
        if "t" in args.get("inputs", "") and args.get("previous_results") is not None:
            result_path = Path(args["previous_results"])
            prediction_file = result_path / f"{run}.pt"
            log.info(f"Prediction_file: {prediction_file}")
            if prediction_file.exists():
                predicted_t = torch.load(prediction_file)
                # If loaded as sequence or extra dims, reduce to single channel/time shape for direct use as input
                if predicted_t.ndim == 4:
                    # common step3 shape from model (1,time,H,W or time,1,H,W)
                    if predicted_t.shape[0] == 1:
                        predicted_t = predicted_t[0]
                    if predicted_t.shape[1] == 1:
                        predicted_t = predicted_t.squeeze(1)
                    if predicted_t.ndim == 3 and predicted_t.shape[0] > 1:
                        predicted_t = predicted_t[-1]
                if predicted_t.ndim == 2:
                    predicted_t = predicted_t.unsqueeze(0)
                # overwrite raw temperature channel
                x["Temperature [C]"] = predicted_t
            else:
                raise FileNotFoundError(f"Step3 prediction not found: {prediction_file}")

        print_bool = False
        loc_hp = load.get_hp_location(x)
        x = transforms(x, loc_hp=loc_hp)
        if info is None:
            calc.add_data(x)
        x = tensor_transform(x)
        y = transforms(y, loc_hp=loc_hp)
        if info is None:
            calc.add_data(y)
        y = tensor_transform(y)
        torch.save(x, args["data_prep"] / "Inputs" / f"{run}.pt")
        torch.save(y, args["data_prep"] / "Labels" / f"{run}.pt")

    if info is not None:
        info["CellsNumberPrior"] = info["CellsNumber"]
        info["PositionHPPrior"] = info["PositionLastHP"]
        assert info["CellsSize"][:2] == cell_size.tolist()[:2], (
            f"Cell size changed between given info.yaml {info['CellsSize']} and data {cell_size.tolist()}"
        )
    else:
        info = dict()
        means = calc.mean()
        stds = calc.std()
        mins = calc.min()
        maxs = calc.max()
        info["Inputs"] = {
            key: {
                "mean": means[key],
                "std": stds[key],
                "min": mins[key],
                "max": maxs[key],
                "norm": "Rescale",
                "index": n,
            }
            for n, key in enumerate(inputs)
        }
        info["Labels"] = {
            key: {
                "mean": means[key],
                "std": stds[key],
                "min": mins[key],
                "max": maxs[key],
                "norm": "Rescale",
                "index": n,
            }
            for n, key in enumerate(outputs)
        }

    info["CellsSize"] = cell_size.tolist()
    assert len(y.shape) == 3, "y is expected to be 2D"
    dims = list(y.shape)[1:]
    info["CellsNumber"] = dims
    info["goal resolution"] = resolution
    try:
        info["PositionLastHP"] = loc_hp.tolist()
    except Exception:
        info["PositionLastHP"] = loc_hp
    save_yaml(info, args["data_prep"] / "info.yaml")
    normalize(args["data_prep"], info, total)
    save_yaml({"dataset": args["data_raw"].name, "inputs": inputs, "outputs": outputs}, args["data_prep"] / "args.yaml")

    return info


def prepare_dataset_for_sequence(args: dict, info: dict = None):
    transforms = get_transforms(inputs=args["inputs"])
    log.info("Transforms: ", [transform.__class__.__name__ for transform in transforms.transforms])
    log.info("Args Inputs:", args["inputs"])

    inputs = expand_property_names(args["inputs"])
    outputs = expand_property_names(args["outputs"])
    times_str = load_time_steps_full_str(Path(args["data_raw"], "RUN_0", "pflotran.h5"))
    times = load_time_steps(Path(args["data_raw"], "RUN_0", "pflotran.h5"))
    time_init = times[0]
    time_prediction = times[1:]

    pflotran_settings = load_yaml(args["data_raw"] / "inputs" / "settings.yaml")
    total_size = np.array(pflotran_settings["grid"]["size"])
    dims = np.array(pflotran_settings["grid"]["ncells"])
    cell_size = total_size / dims
    resolution = cell_size[0]

    if info is None:
        calc = WelfordStatistics()
    tensor_transform = ToTensorTransform()
    data_paths, runs = load.detect_datapoints(args["data_raw"])
    total = len(data_paths)
    print_bool = False
    for data_path, run in tqdm(zip(data_paths, runs), desc="Converting", total=total):
        x = load.load_raw_data(data_path, times_str[0], inputs, dims, times_str[1], print_bool=print_bool)

        # For step4: if temperature is requested in inputs, replace with step3 prediction
        if "t" in args.get("inputs", "") and args.get("previous_results") is not None:
            result_path = Path(args["previous_results"])
            prediction_file = result_path / f"{run}.pt"
            if prediction_file.exists():
                predicted_t = torch.load(prediction_file)
                # If loaded as sequence or extra dims, reduce to single spatial channel
                if predicted_t.ndim == 4:
                    if predicted_t.shape[0] == 1:
                        predicted_t = predicted_t[0]
                    if predicted_t.ndim == 3 and predicted_t.shape[0] == 1:
                        predicted_t = predicted_t.squeeze(0)
                    if predicted_t.ndim == 3 and predicted_t.shape[0] > 1:
                        predicted_t = predicted_t[-1]
                if predicted_t.ndim == 2:
                    predicted_t = predicted_t.unsqueeze(0)
                log.info(f"Shape of predicted_t: {predicted_t.size()}")
                x["Temperature [C]"] = predicted_t
            else:
                raise FileNotFoundError(f"Step3 prediction not found: {prediction_file}")

        y = {output: [] for output in outputs}
        for time in times_str:
            for output in outputs:
                log.info(f"times_str[1:]: {times_str[1:]}")
                value = load.load_raw_data(data_path, time, [output], dims, times_str[1:], print_bool=print_bool)[
                    output
                ]
                value = torch.unsqueeze(value, dim=0)  # (channels, time, H, W)
                y[output].append(value)

        for output in outputs:
            y[output] = torch.cat(y[output], dim=0)

        print_bool = False

        loc_hp = load.get_hp_location(x)
        x = transforms(x, loc_hp=loc_hp)
        if info is None:
            calc.add_data(x)
        x = tensor_transform(x)
        x = torch.unsqueeze(x, dim=1)  # add time dim -> (C, T, H, W)
        y = transforms(y, loc_hp=loc_hp)
        if info is None:
            calc.add_data(y)
        y = tensor_transform(y)
        torch.save(x, args["data_prep"] / "Inputs" / f"{run}.pt")
        torch.save(y, args["data_prep"] / "Labels" / f"{run}.pt")

    if info is not None:
        info["CellsNumberPrior"] = info["CellsNumber"]
        info["PositionHPPrior"] = info["PositionLastHP"]
        assert info["CellsSize"][:2] == cell_size.tolist()[:2], (
            f"Cell size changed between given info.yaml {info['CellsSize']} and data {cell_size.tolist()}"
        )
    else:
        info = dict()
        means = calc.mean()
        stds = calc.std()
        mins = calc.min()
        maxs = calc.max()
        info["Inputs"] = {
            key: {
                "mean": means[key],
                "std": stds[key],
                "min": mins[key],
                "max": maxs[key],
                "norm": "Rescale",
                "index": n,
            }
            for n, key in enumerate(inputs)
        }
        info["Labels"] = {
            key: {
                "mean": means[key],
                "std": stds[key],
                "min": mins[key],
                "max": maxs[key],
                "norm": "Rescale",
                "index": n,
            }
            for n, key in enumerate(outputs)
        }

    info["CellsSize"] = cell_size.tolist()
    dims = list(y.shape)[-2:]
    info["CellsNumber"] = dims
    info["goal resolution"] = resolution
    try:
        info["PositionLastHP"] = loc_hp.tolist()
    except Exception:
        info["PositionLastHP"] = loc_hp
    save_yaml(info, args["data_prep"] / "info.yaml")
    normalize(args["data_prep"], info, total)
    save_yaml({"dataset": args["data_raw"].name, "inputs": inputs, "outputs": outputs}, args["data_prep"] / "args.yaml")

    return info


def expand_property_names(properties: str):
    translation = {
        "x": "Liquid X-Velocity [m_per_y]",
        "y": "Liquid Y-Velocity [m_per_y]",
        "z": "Liquid Z-Velocity [m_per_y]",
        "p": "Liquid Pressure [Pa]",
        "k": "Permeability X [m^2]",
        "g": "Pressure Gradient [-]",
        "i": "Material ID",
        "t": "Temperature [C]",
        "s": "SDF",
        "l": "Line Integral Convolution",
        "d": "Streamlines Faded [-]",
        "c": "Streamlines Faded Outer [-]",
        "1": "Streamline-Sum_Position",
        "2": "Streamline-Sum_RelativeUncertainty",
        "3": "Streamline-Sum_TimeFaded-Position",
        "4": "Streamline-Max_TimeFaded",
        "5": "Streamline-Sum_TimeSeasons-Position",
        "6": "Streamline-Max_TimeSeasons",
        "7": "Streamline-TemperatureApproximation",
    }
    possible_vars = ",".join(translation.keys())
    assert all((prop in possible_vars) for prop in properties), (
        f"input parameters have to be a string of characters, each of which is either {possible_vars}"
    )
    return [translation[prop] for prop in properties]
