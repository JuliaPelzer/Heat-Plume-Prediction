import logging
import os
import argparse
from main import run_experiment
from data.utils import load_settings, save_settings

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)        # level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_datasets", type=str, default="/home/pelzerja/pelzerja/test_nn/datasets")
    parser.add_argument("--dataset_name", type=str, default="benchmark_dataset_2d_100datapoints") #"dataset3D_100dp_perm_vary" #"dataset3D_100dp_perm_iso" #
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--overfit", type=bool, default=False)
    parser.add_argument("--epochs", type=int, default=40000)

    args = parser.parse_args()
    kwargs = load_settings(".", "settings_training") # TODO

    kwargs["path_to_datasets"] = args.path_to_datasets
    kwargs["dataset_name"] = args.dataset_name
    kwargs["device"] = args.device
    kwargs["lr"]=args.lr
    kwargs["overfit"] = args.overfit
    kwargs["n_epochs"] = args.epochs
    if kwargs["overfit"]:
        overfit_str = "overfit_"
    else:
        overfit_str = ""
    input_combis = ["pk", "t", "px", "py", "xy"]
    for model in ["unet"]: #, "fc"]:
        kwargs["model_choice"] = model
        for input in input_combis:
            kwargs["inputs"] = input
            kwargs["name_folder_destination"] = "try" #f"{kwargs['model_choice']}_{overfit_str}epochs_{kwargs['n_epochs']}_inputs_{kwargs['inputs']}_{kwargs['dataset_name']}_timestep0"
            try:
                os.mkdir(os.path.join(os.getcwd(), "runs", kwargs["name_folder_destination"]))
            except FileExistsError:
                pass
            save_settings(kwargs, os.path.join(os.getcwd(), "runs", kwargs["name_folder_destination"]), "settings_training")
            run_experiment(**kwargs)
    
    #tensorboard --logdir runs/