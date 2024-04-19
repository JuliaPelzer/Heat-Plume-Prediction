from pathlib import Path
import yaml

def assertions_args(args):
    if args.case in ["test", "finetune"]:
        assert args.model is not None, "Model name required for testing or finetuning"
    else:
        assert args.model is None, "Model name should not be defined for training"

def make_paths(args):
    dir_file = "paths.yaml"
    if not Path(dir_file).exists():
        raise FileNotFoundError(f"{dir_file} not found")
    
    paths = load_yaml(dir_file)

    # dataset_raw, dataset_prep
    args.data_raw = Path(paths["default_raw_dir"]) / args.problem / args.data_raw
    if not args.data_raw.exists():
        raise FileNotFoundError(f"{args.data_raw} not found")
    
    if args.data_prep is None:
        args.data_prep = args.data_raw.name + " inputs_" + args.inputs

    args.data_prep = Path(paths["datasets_prepared_dir"]) / args.problem / args.data_prep
    args.data_prep.mkdir(parents=True, exist_ok=True)
    (args.data_prep / "Inputs").mkdir(parents=True, exist_ok=True)
    (args.data_prep / "Labels").mkdir(parents=True, exist_ok=True)

    # model, destination
    if args.destination is None:
        args.destination = args.data_prep.name + " box"+str(args.len_box) + " skip"+str(args.skip_per_dir)
    if args.case == "train":
        args.destination = Path(paths["models_1hp_dir"]) / args.problem / args.destination
        args.destination.mkdir(parents=True, exist_ok=True)
        args.model = args.destination
    else:
        args.model = Path(paths["models_1hp_dir"]) / args.problem / args.model
        if not (args.model / "model.pt").exists() or not (args.model / "info.yaml").exists():
            raise FileNotFoundError(f"model.pt or info.yaml not found in '{args.model.name}'")
        args.destination = args.model / (args.destination + " " + args.case)
        args.destination.mkdir(parents=True, exist_ok=True)

def save_notes(args):
    if args.notes is not None:
        with open(args.destination / "notes.txt", "w") as file:
            file.write(args.notes)

def save_cla(args):
    with open(args.destination / "command_line_arguments.yaml", "w") as file:
        tmp = vars(args).copy()
        for arg in vars(args):
            # if arg a Path object, convert to string
            if isinstance(vars(args)[arg], Path):
                tmp[arg] = str(vars(args)[arg])
        yaml.dump(tmp, file)

# OTHER UTILS
def is_empty(path:Path):
    return not bool(list(path.iterdir()))

def load_yaml(file):
    with open(file, "r") as f:
        data = yaml.safe_load(f)
    return data

def save_yaml(file, data):
    with open(file, "w") as f:
        yaml.dump(data, f)
