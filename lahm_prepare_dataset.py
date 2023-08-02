import argparse
import logging
import os
import pathlib
import yaml

from data.utils import SettingsTraining
from prepare_dataset import prepare_dataset

def set_paths(dataset_name: str, name_extension: str = None):
    if os.path.isfile("paths.yaml"):
        with open("paths.yaml", "r") as f:
            paths = yaml.safe_load(f)
            default_raw_dir = paths["default_raw_dir"]
            datasets_prepared_dir = paths["datasets_prepared_dir"]
    else:
        if os.path.exists("/scratch/sgs/pelzerja/"):
            default_raw_dir = "/scratch/sgs/pelzerja/datasets/1hp_boxes"
            datasets_prepared_dir="/home/pelzerja/pelzerja/test_nn/datasets_prepared/1HP_NN" # TODO CHANGE BACK TO 1HP_NN
    
    dataset_prepared_path = os.path.join(datasets_prepared_dir, dataset_name+name_extension)

    return default_raw_dir, datasets_prepared_dir, dataset_prepared_path

if __name__ == "__main__":
    # level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    logging.basicConfig(level=logging.WARNING)
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="benchmark_dataset_2d_100datapoints")
    # benchmark_dataset_2d_20dp_2hps benchmark_testcases_4 benchmark_dataset_2d_100dp_vary_hp_loc benchmark_dataset_2d_100datapoints dataset3D_100dp_perm_vary dataset3D_100dp_perm_iso
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_choice", type=str, default="lahm")
    parser.add_argument("--inputs_prep", type=str, default="pksi")
    parser.add_argument("--name_folder_destination", type=str, default="")
    parser.add_argument("--name_extension", type=str, default="_lahm")
    args = parser.parse_args()
    
    default_raw_dir, datasets_prepared_dir, dataset_prepared_full_path = set_paths(args.dataset_name, args.name_extension)
    args.datasets_path = datasets_prepared_dir
    args.case = "test"
    args.epochs = 0
    args.path_to_model = ""

    # prepare dataset if not done yet
    if not os.path.exists(dataset_prepared_full_path):
        prepare_dataset(raw_data_directory = default_raw_dir,
                        datasets_path = datasets_prepared_dir,
                        dataset_name = args.dataset_name,
                        input_variables = args.inputs_prep,
                        name_extension=args.name_extension,)
        print(f"Dataset {dataset_prepared_full_path} prepared")
    else:
        print(f"Dataset {dataset_prepared_full_path} already prepared")
    args.dataset_name += args.name_extension

    settings = SettingsTraining(**vars(args))
    if settings.name_folder_destination == "":
        settings.name_folder_destination = f"current_{settings.model_choice}_{settings.dataset_name}"
    destination_dir = pathlib.Path(os.getcwd(), "runs", settings.name_folder_destination)
    destination_dir.mkdir(parents=True, exist_ok=True)

    settings.save()

    # beep()
    # tensorboard --logdir=runs/ --host localhost --port 8088
