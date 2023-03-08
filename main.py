import datetime as dt
import sys
import logging
import numpy as np
import os
import argparse
from torch.nn import MSELoss
from networks.models import create_model
from networks.losses import MSELossExcludeNotChangedTemp, InfinityLoss, normalize
from torch import cuda, save
from data.dataset_loading import init_data
from data.utils import load_settings, save_settings
from solver import Solver
from visualization.visualize_data import plot_sample
from utils.utils_networks import count_parameters, append_results_to_csv


def run_experiment(n_epochs: int = 1000, lr: float = 5e-3,
    inputs: str = "pk", model_choice: str="unet",
    name_folder_destination: str = "default", dataset_name: str = "small_dataset_test",
    path_to_datasets: str = "/home/pelzerja/Development/simulation_groundtruth_pflotran/Phd_simulation_groundtruth/datasets",
    overfit: bool = True, device: str = None):
    time_begin = dt.datetime.now()

    # parameters of data
    reduce_to_2D = True
    reduce_to_2D_xy = True
    overfit = overfit
    # init data
    datasets_2D, dataloaders_2D = init_data(
        dataset_name=dataset_name,
        path_to_datasets=path_to_datasets,
        batch_size=100,
        reduce_to_2D=reduce_to_2D,
        reduce_to_2D_xy=reduce_to_2D_xy,
        inputs=inputs,
        labels="t",
        overfit=overfit,
    )

    # model choice
    in_channels = len(inputs) + 1
    model = create_model(model_choice, in_channels, datasets_2D, reduce_to_2D)

    if device is None:
        device = "cuda" if cuda.is_available() else "cpu"
    logging.warning(f"Using {device} device")
    model.to(device)

    number_parameter = count_parameters(model)
    logging.info(f"Model {model_choice} with number of parameters: {number_parameter}")

    # parameters of training
    loss_fn_str = "MSE"
    if loss_fn_str == "ExcludeNotChangedTemp":
        ignore_temp = 10.6
        temp_mean, temp_std = (
            dataloaders_2D["train"].dataset.mean_labels["Temperature [C]"],
            dataloaders_2D["train"].dataset.std_labels["Temperature [C]"],
        )
        normalized_ignore_temp = normalize(ignore_temp, temp_mean, temp_std)
        print(temp_std, temp_mean, normalized_ignore_temp)
        loss_fn = MSELossExcludeNotChangedTemp(ignore_temp=normalized_ignore_temp)
    elif loss_fn_str == "MSE":
        loss_fn = MSELoss()
    elif loss_fn_str == "Infinity":
        loss_fn = InfinityLoss()

    n_epochs = n_epochs
    lr = float(lr)

    # training
    if overfit:
        solver = Solver(
            model,
            dataloaders_2D["train"],
            dataloaders_2D["train"],
            learning_rate=lr,
            loss_func=loss_fn,
        )
    else:
        solver = Solver(
            model,
            dataloaders_2D["train"],
            dataloaders_2D["val"],
            learning_rate=lr,
            loss_func=loss_fn,
        )
    try:
        solver.train(
            device, n_epochs=n_epochs, name_folder=name_folder_destination
        )
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        pass

    # visualization
    if overfit:
        error, error_mean, final_max_error = plot_sample(
            model,
            dataloaders_2D["train"],
            device,
            name_folder_destination,
            plot_name=name_folder_destination + "/plot_train_sample_applied",
        )
    else:
        error, error_mean, final_max_error = plot_sample(
            model,
            dataloaders_2D["train"],
            device,
            name_folder_destination,
            plot_name=name_folder_destination + "/plot_train_sample_applied",
            amount_plots=3,
        )
        error, error_mean, final_max_error = plot_sample(
            model,
            dataloaders_2D["val"],
            device,
            name_folder_destination,
            plot_name=name_folder_destination + "/plot_val_sample_applied",
            amount_plots=3,
        )

    # save model
    save(model.state_dict(), os.path.join(os.getcwd(),"runs",name_folder_destination, f"{model_choice}_{inputs}.pt"))

    time_end = dt.datetime.now()
    duration = f"{(time_end-time_begin).seconds//60} minutes {(time_end-time_begin).seconds%60} seconds"
    print(f"Time needed for experiment: {duration}")

    # logging
    results = {
        "timestamp": time_begin,
        "model": model_choice,
        "dataset": dataset_name,
        "overfit": overfit,
        "inputs": inputs,
        "n_epochs": n_epochs,
        "lr": lr,
        "error_mean": error_mean[-1],
        "error_max": final_max_error,
        "duration": duration,
        "name_destination_folder": name_folder_destination,
    }
    append_results_to_csv(results, "runs/collected_results_rough_idea.csv")

    model.to("cpu")
    # del model
    return model

if __name__ == "__main__":
    
    logging.basicConfig(level=logging.WARNING)        # level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    
    parser = argparse.ArgumentParser()
    kwargs = load_settings(".", "settings_training")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="benchmark_dataset_2d_100datapoints") #"dataset3D_100dp_perm_vary" #"dataset3D_100dp_perm_iso" #
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--overfit", type=bool, default=False)
    parser.add_argument("--epochs", type=int, default=40000)

    args = parser.parse_args()
    kwargs = load_settings(".", "settings_training") # TODO

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