from pathlib import Path
import yaml
import numpy as np
import torch

from preprocessing.transforms import NormalizeTransform
from utils.utils_args import is_empty, load_yaml, save_yaml

def preprocessing(args:dict):
    assert not is_unprepared(args["data_prep"]), f"For benchmark case I currently expect the data to be prepared!"
    print(f"Dataset: {args['data_prep']}")

    if args["case"] == "train":
        info = load_yaml(args["data_prep"]/"info.yaml") 
        save_yaml(info, args["destination"]/"info.yaml")

# helper function
def is_unprepared(path:Path):
    return is_empty(path / "Inputs") or is_empty(path / "Labels") or not (path / "info.yaml").exists()

def import_dataset(path_orig: Path, path_desti:Path):
    path_desti.mkdir(parents=True, exist_ok=True)
    (path_desti / "Inputs").mkdir(parents=True, exist_ok=True)
    (path_desti / "Labels").mkdir(parents=True, exist_ok=True)

    # get and extend norm info
    norm_info = yaml.safe_load(open(path_orig / "general" / "normalization_info.yaml", "r"))
    gen_info = yaml.safe_load(open(path_orig / "general" / "dataset_info.yaml", "r"))
    norm_info["CellsSize"] = [float(gen_info["spatial domain"]["res_x"]), float(gen_info["spatial domain"]["res_y"])]
    nameT = "Temperature (timedependent) [degree C]"
    temp_info = norm_info["Labels"][nameT]
    norm_info["Labels"] = {f"{nameT} Summer": temp_info.copy(),
                        f"{nameT} Autumn": temp_info.copy(),
                        f"{nameT} Winter": temp_info.copy(),
                        f"{nameT} Spring": temp_info.copy()}
    norm_info["Labels"][f"{nameT} Summer"]["index"] = 0
    norm_info["Labels"][f"{nameT} Autumn"]["index"] = 1
    norm_info["Labels"][f"{nameT} Winter"]["index"] = 2
    norm_info["Labels"][f"{nameT} Spring"]["index"] = 3
    yaml.safe_dump(norm_info, open(path_desti / "info.yaml", "w"))

    norm = NormalizeTransform(norm_info)

    # copy temperature timeseries over
    temp_series = np.load(path_orig / "general/temperature_injection_series.npy")
    np.save(path_desti / "temperature_injection_series.npy", temp_series)

    # get numpy data and save as torch tensors
    for i in path_orig.glob("training_data/Sim_*.npz"):
        name = i.stem
        data = np.load(path_orig / "training_data" / f"{name}.npz")
        inputs = torch.from_numpy(data["inputs"])
        labels = torch.from_numpy(data["labels"][-4:])
        print(name, inputs.shape, labels.shape)
        inputs = norm(inputs, "Inputs")
        labels = norm(labels, "Labels")
        inputs = inputs.to(torch.float32)
        labels = labels.to(torch.float32)
        torch.save(inputs, path_desti / "Inputs" / f"{name}.pt")
        torch.save(labels, path_desti / "Labels" / f"{name}.pt")

if __name__ == "__main__":
    path_orig = Path("/scratch/sgs/pelzerja/datasets/bm/lorentz/data/step3")
    path_desti = Path("/scratch/sgs/pelzerja/datasets_prepared/bm/step3")
    import_dataset(path_orig, path_desti)