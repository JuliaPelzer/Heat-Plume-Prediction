import datetime as dt
import logging
import os
import argparse
from networks.models import create_model, load_model
from networks.losses import create_loss_fn
from torch import cuda, save
from data.dataset_loading import init_data, make_dataset_for_test
from data.utils import load_settings, save_settings
from solver import Solver
from visualization.visualize_data import plot_sample
from utils.utils_networks import count_parameters, append_results_to_csv

def run_training(n_epochs: int = 1000, lr: float = 5e-3, inputs: str = "pk", model_choice: str="unet", name_folder_destination: str = "default", dataset_name: str = "small_dataset_test",
    path_to_datasets: str = "/home/pelzerja/Development/simulation_groundtruth_pflotran/Phd_simulation_groundtruth/datasets", overfit: bool = True, device: str = None):
    time_begin = dt.datetime.now()

    # parameters of data
    reduce_to_2D = True
    reduce_to_2D_xy = True
    sdf = True
    # init data
    datasets, dataloaders = init_data(dataset_name=dataset_name, path_to_datasets=path_to_datasets,
        batch_size=100, sdf=sdf, reduce_to_2D=reduce_to_2D, reduce_to_2D_xy=reduce_to_2D_xy, inputs=inputs, labels="t", overfit=overfit, name_folder_destination=name_folder_destination,)

    if device is None:
        device = "cuda" if cuda.is_available() else "cpu"
    logging.warning(f"Using {device} device")

    # model choice
    in_channels = len(inputs) + 1
    model = create_model(model_choice, in_channels, datasets, reduce_to_2D)
    model.to(device)

    number_parameter = count_parameters(model)
    logging.warning(f"Model {model_choice} with number of parameters: {number_parameter}")

    train = True
    if train:
        # parameters of training
        loss_fn_str = "MSE"
        loss_fn = create_loss_fn(loss_fn_str, dataloaders)
        n_epochs = n_epochs
        lr = float(lr)
        # training
        if overfit:
            solver = Solver(model,dataloaders["train"],dataloaders["train"],learning_rate=lr,loss_func=loss_fn,)
        else:
            solver = Solver(model,dataloaders["train"],dataloaders["val"],learning_rate=lr,loss_func=loss_fn,)
        try:
            solver.load_lr_schedule(os.path.join(os.getcwd(), "runs", name_folder_destination, "learning_rate_history.csv"))
            solver.train(device, n_epochs=n_epochs, name_folder=name_folder_destination)
        except KeyboardInterrupt:
            logging.warning("KeyboardInterrupt")
    else:
        # load model
        model = load_model({"model_choice":model_choice, "in_channels":in_channels}, os.path.join(os.getcwd(), "runs", name_folder_destination), "model")
        model.to(device)
    # save model
    if train:
        save(model.state_dict(), os.path.join(os.getcwd(), "runs", name_folder_destination, "model.pt"))
        solver.save_lr_schedule(os.path.join(os.getcwd(), "runs", name_folder_destination, "learning_rate_history.csv"))
        
    # visualization
    if True:
        if overfit:
            _, error_mean, final_max_error = plot_sample(model, dataloaders["train"], device, name_folder_destination, plot_name=name_folder_destination + "/plot_train_sample",)
        else:
            _, error_mean, final_max_error = plot_sample(model, dataloaders["train"], device, name_folder_destination, plot_name=name_folder_destination + "/plot_train_sample", amount_plots=2,)
            _, error_mean, final_max_error = plot_sample(model, dataloaders["val"], device, name_folder_destination, plot_name=name_folder_destination + "/plot_val_sample", amount_plots=5,)

    time_end = dt.datetime.now()
    duration = f"{(time_end-time_begin).seconds//60} minutes {(time_end-time_begin).seconds%60} seconds"
    print(f"Time needed for experiment: {duration}")

    # logging
    results = {"timestamp": time_begin, "model": model_choice, "dataset": dataset_name, "overfit": overfit, "inputs": inputs, "n_epochs": n_epochs, "lr": lr, "error_mean": error_mean[-1], "error_max": final_max_error, "duration": duration, "name_destination_folder": name_folder_destination,}
    append_results_to_csv(results, "runs/collected_results_rough_idea.csv")

    model.to("cpu")
    return model

def run_tests(inputs: str = "pk", model_choice: str="unet", name_folder_destination: str = "default", dataset_name: str = "small_dataset_test",
    path_to_datasets: str = "/home/pelzerja/Development/simulation_groundtruth_pflotran/Phd_simulation_groundtruth/datasets", device: str = None,
    n_epochs: int = 1000, lr: float = 5e-3, overfit: bool = True,):

    # init data
    _, dataloader = make_dataset_for_test(dataset_name=dataset_name, path_to_datasets=path_to_datasets, inputs=inputs, name_folder_destination=name_folder_destination,)

    # model choice
    in_channels = len(inputs) + 1
    model = create_model(model_choice, in_channels)
    model.to(device)

    number_parameter = count_parameters(model)
    logging.warning(f"Model {model_choice} with number of parameters: {number_parameter}")

    # load model
    model = load_model({"model_choice":model_choice, "in_channels":in_channels}, os.path.join(os.getcwd(), "runs", name_folder_destination), "model")
    model.to(device)
        
    # visualization
    time_begin = dt.datetime.now()
    plot_sample(model, dataloader, device, name_folder_destination, plot_name=name_folder_destination + "/plot_TEST_sample",)

    time_end = dt.datetime.now()
    duration = f"{(time_end-time_begin).seconds//60} minutes {(time_end-time_begin).seconds%60} seconds"
    print(f"Time needed for only inference and plotting: {duration}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)        # level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    
    remote = True
    parser = argparse.ArgumentParser()
    if remote:
        parser.add_argument("--path_to_datasets", type=str, default="/home/pelzerja/pelzerja/test_nn/datasets")
    else:
        parser.add_argument("--path_to_datasets", type=str, default="/home/pelzerja/Development/simulation_groundtruth_pflotran/Phd_simulation_groundtruth/datasets")
    
    parser.add_argument("--dataset_name", type=str, default="dataset3D_100dp_perm_iso") # benchmark_testcases_3 benchmark_dataset_2d_100datapoints dataset3D_100dp_perm_vary dataset3D_100dp_perm_iso
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--overfit", type=bool, default=False)
    parser.add_argument("--epochs", type=int, default=40000)

    args = parser.parse_args()
    kwargs = load_settings(".", "settings_training")

    kwargs["path_to_datasets"] = args.path_to_datasets
    kwargs["dataset_name"] = args.dataset_name
    kwargs["device"] = args.device
    kwargs["lr"]=args.lr
    kwargs["overfit"] = args.overfit
    kwargs["n_epochs"] = args.epochs
    input_combis = ["pk"] #, "xy", "pky"] 
    for model in ["unet"]: #, "fc"]:
        kwargs["model_choice"] = model
        for input in input_combis:
            kwargs["inputs"] = input
            # kwargs["name_folder_destination"] = "crossVal_benchmark_to_varyPerm"
            kwargs["name_folder_destination"] = f"current_{kwargs['model_choice']}_{kwargs['dataset_name']}_inputs_{kwargs['inputs']}"
            try:
                os.mkdir(os.path.join(os.getcwd(), "runs", kwargs["name_folder_destination"]))
            except FileExistsError:
                pass
            save_settings(kwargs, os.path.join(os.getcwd(), "runs", kwargs["name_folder_destination"]), "settings_training")
            run_training(**kwargs)
            # run_tests(**kwargs)
    
    #tensorboard --logdir runs/