import datetime as dt
import logging
import os
import argparse
from networks.models import create_model, load_model
from networks.losses import create_loss_fn
from torch import cuda, save
from data.utils import SettingsTraining
from solver import Solver
from utils.visualize_data import plot_sample
from utils.utils_networks import count_parameters, append_results_to_csv
from data.dataset import SimulationDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch


def run_training(settings: SettingsTraining):
    time_begin = dt.datetime.now()

    # init data
    dataset = SimulationDataset(
        os.path.join(settings.datasets_path, settings.dataset_name))
    generator = torch.Generator().manual_seed(1)
    datasets = random_split(
        dataset, [0.8, 0.1, 0.1], generator=generator)

    dataloaders = {
        "train": DataLoader(
            datasets[0], batch_size=100, shuffle=True, num_workers=0),
        "val": DataLoader(
            datasets[1], batch_size=100, shuffle=True, num_workers=0),
        "test": DataLoader(
            datasets[2], batch_size=100, shuffle=True, num_workers=0)
    }

    if settings.device is None:
        settings.device = "cuda" if cuda.is_available() else "cpu"
    logging.warning(f"Using {settings.device} device")

    # model choice
    in_channels = dataset[0][0].shape[0]
    if not settings.finetune:
        model = create_model(settings.model_choice, in_channels)
    else:
        model = load_model({"model_choice": settings.model_choice,
                           "in_channels": in_channels}, settings.path_to_model)
    model.to(settings.device)

    number_parameter = count_parameters(model)
    logging.warning(
        f"Model {settings.model_choice} with number of parameters: {number_parameter}")

    train = True
    if train:
        # parameters of training
        loss_fn_str = "MSE"
        loss_fn = create_loss_fn(loss_fn_str, dataloaders)
        # training
        solver = Solver(model, dataloaders["train"], dataloaders["val"],
                        loss_func=loss_fn, finetune=settings.finetune)
        try:
            solver.load_lr_schedule(os.path.join(os.getcwd(
            ), "runs", settings.name_folder_destination, "learning_rate_history.csv"))
            solver.train(settings)
        except KeyboardInterrupt:
            logging.warning("Stopping training early")
    else:
        # load model
        model = load_model({"model_choice": settings.model_choice, "in_channels": in_channels}, os.path.join(
            os.getcwd(), "runs", settings.name_folder_destination), "model")
        model.to(settings.device)
    # save model
    if train:
        save(model.state_dict(), os.path.join(os.getcwd(), "runs",
             settings.name_folder_destination, "model.pt"))
        solver.save_lr_schedule(os.path.join(os.getcwd(
        ), "runs", settings.name_folder_destination, "learning_rate_history.csv"))

    # visualization
    if True:
        error_mean, final_max_error = plot_sample(
            model, dataloaders["val"], settings.device, plot_name=settings.name_folder_destination + "/plot_val_sample", amount_plots=10,)
        error_mean, final_max_error = plot_sample(
            model, dataloaders["train"], settings.device, plot_name=settings.name_folder_destination + "/plot_train_sample", amount_plots=2,)

    time_end = dt.datetime.now()
    duration = f"{(time_end-time_begin).seconds//60} minutes {(time_end-time_begin).seconds%60} seconds"
    print(f"Experiment took {duration}")

    # logging
    results = {"timestamp": time_begin, "model": settings.model_choice, "dataset": settings.dataset_name, "n_epochs": settings.epochs,
               "error_mean": error_mean[-1], "error_max": final_max_error, "duration": duration, "name_destination_folder": settings.name_folder_destination, }
    append_results_to_csv(results, "runs/collected_results_rough_idea.csv")

    model.to("cpu")
    return model


def run_tests(settings: SettingsTraining):

    # init data
    _, dataloader = make_dataset_for_test(settings)

    # model choice
    in_channels = len(settings.inputs)
    model = create_model(settings.model_choice, in_channels)
    model.to(settings.device)

    number_parameter = count_parameters(model)
    logging.warning(
        f"Model {settings.model_choice} with number of parameters: {number_parameter}")

    # load model
    model = load_model({"model_choice": settings.model_choice, "in_channels": in_channels}, os.path.join(
        os.getcwd(), "runs", settings.name_folder_destination), "model")
    model.to(settings.device)

    # visualization
    time_begin = dt.datetime.now()
    plot_sample(model, dataloader, settings.device,
                plot_name=settings.name_folder_destination + "/plot_TEST_sample", amount_plots=10,)

    time_end = dt.datetime.now()
    duration = f"{(time_end-time_begin).seconds//60} minutes {(time_end-time_begin).seconds%60} seconds"
    print(f"Time needed for only inference and plotting: {duration}")


if __name__ == "__main__":
    # level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser()

    # benchmark_dataset_2d_20dp_2hps benchmark_testcases_4 benchmark_dataset_2d_100dp_vary_hp_loc benchmark_dataset_2d_100datapoints dataset3D_100dp_perm_vary dataset3D_100dp_perm_iso
    parser.add_argument("--datasets_path", type=str)
    parser.add_argument("--dataset_name", type=str,
                        default="benchmark_dataset_2d_100dp_vary_perm")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--epochs", type=int, default=30000)
    parser.add_argument("--finetune", type=bool, default=False)
    parser.add_argument("--path_to_model", type=str,
                        default="benchmarkPLUSdataset_2d_100dp_vary_hp_loc/unet_inputs_pk_MatID_noPowerOf2")
    parser.add_argument("--model_choice", type=str, default="unet")
    parser.add_argument("--name_folder_destination",
                        type=str, default="default")
    args = parser.parse_args()

    settings = SettingsTraining(**vars(args))
    settings.name_folder_destination = f"current_{settings.model_choice}_{settings.dataset_name}"

    try:
        os.mkdir(os.path.join(os.getcwd(), "runs",
                 settings.name_folder_destination))
    except FileExistsError:
        pass
    settings.save()

    run_training(settings)
    # run_tests(settings)
    # tensorboard --logdir runs/
