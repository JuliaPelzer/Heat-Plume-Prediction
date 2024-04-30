from pathlib import Path
import yaml
from typing import List, Union
import argparse

def assertions_args(args:dict):
    if args["case"] in ["test", "finetune"]:
        assert args["model"] is not None, "Model name required for testing or finetuning"
    else:
        assert args["model"] is None, "Model name should not be defined for training"

def make_paths(args:dict, make_model_and_destination_bool:bool=True):
    paths = get_paths(name="paths.yaml")
    get_raw_path(args, Path(paths["default_raw_dir"]))
    make_prep_path(args, prep_dir=Path(paths["datasets_prepared_dir"]))
    if make_model_and_destination_bool:
        make_model_and_destination_paths(args, Path(paths["models_1hp_dir"]))

def get_paths(name:str="paths.yaml"):
    if not Path(name).exists():
        raise FileNotFoundError(f"{name} not found in cwd")
    paths = load_yaml(name)
    return paths

def get_raw_path(args:dict, raw_dir: Path):
    # dataset_raw
    args["data_raw"] = raw_dir / args["problem"] / args["data_raw"]
    if not args["data_raw"].exists():
        raise FileNotFoundError(f"{args['data_raw']} not found")
    
def make_prep_path(args:dict, prep_dir: Path):
    # dataset_prep
    if args["data_prep"] is None:
        args["data_prep"] = args["data_raw"].name + " inputs_" + args["inputs"]

    args["data_prep"] = prep_dir / args["problem"] / args["data_prep"]
    args["data_prep"].mkdir(parents=True, exist_ok=True)
    (args["data_prep"] / "Inputs").mkdir(parents=True, exist_ok=True)
    (args["data_prep"] / "Labels").mkdir(parents=True, exist_ok=True)

def make_model_and_destination_paths(args:dict, models_dir: Path):
    # model, destination
    if args["destination"] is None:
        args["destination"] = args["data_prep"].name + " box"+str(args["len_box"]) + " skip"+str(args["skip_per_dir"])
    if args["case"] == "train":
        args["destination"] = models_dir / args["problem"] / args["destination"]
        args["destination"].mkdir(parents=True, exist_ok=True)
        args["model"] = args["destination"]
    else:
        args["model"] = models_dir / args["problem"] / args["model"]
        if not (args["model"] / "model.pt").exists() or not (args["model"] / "info.yaml").exists():
            raise FileNotFoundError(f"model.pt or info.yaml not found in {args['model'].name}")
        args["destination"] = args["model"] / (args["destination"] + " " + args["case"])
        args["destination"].mkdir(parents=True, exist_ok=True)

def save_notes(args:dict):
    if args["notes"] is not None:
        with open(args["destination"] / "notes.txt", "w") as file:
            file.write(args["notes"])

def load_yaml(path: Path, **kwargs) -> dict:
    with open(path, "r") as file:
        try:
            args = yaml.safe_load(file, **kwargs)
        except:
            args = yaml.load(file, **kwargs)
    return args

def save_yaml(args:dict, destination_file):
    with open(destination_file, "w") as file:
        try:
            tmp = args.copy()
            for arg in vars(args):
                try:
                    for info in vars(arg):
                        # if arg a Path object, convert to string
                        if isinstance(vars(arg)[info], Path):
                            tmp[info] = str(vars(arg)[info])
                except:
                    # if arg a Path object, convert to string
                    if isinstance(vars(args)[arg], Path):
                        tmp[arg] = str(vars(args)[arg])
            yaml.dump(tmp, file)
        except:
            yaml.dump(args, file)

def get_run_ids_from_prep(dir: Path) -> List[int]:
    run_ids = []
    for file in dir.iterdir():
        if file.suffix == ".pt":
            run_ids.append(int(file.stem.split("_")[-1]))
            # print(f"Found run_id {run_ids[-1]}")
    run_ids.sort()
    return run_ids

def get_run_ids_from_raw(dir: Path) -> List[int]:
    run_ids = []
    for folder in dir.iterdir():
        if folder.is_dir() and folder.stem.startswith("RUN"):
            run_ids.append(int(folder.stem.split("_")[-1]))
            # print(f"Found run_id {run_ids[-1]}")
    run_ids.sort()
    return run_ids

# OTHER UTILS
def is_empty(path:Path):
    return not bool(list(path.iterdir()))