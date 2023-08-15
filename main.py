import argparse
import datetime as dt
import logging
import os
import pathlib
import time
import yaml

import torch
from torch import cuda, save
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from data.dataset import SimulationDataset, _get_splits
from data.utils import SettingsTraining, SettingsPrepare
from networks.losses import create_loss_fn
from networks.models import compare_models, create_model, load_model
from prepare_dataset import prepare_dataset
from solver import Solver
from utils.utils import beep, set_paths
from utils.utils_networks import append_results_to_csv, count_parameters
from utils.visualize_data import error_measurements, plot_sample
# from torchsummary import summary


def init_data(settings: SettingsTraining, seed=1):
    dataset = SimulationDataset(os.path.join(settings.datasets_path, settings.dataset_name))
    generator = torch.Generator().manual_seed(seed)

    if settings.case in ["train", "finetune"]:
        datasets = random_split(
            dataset, _get_splits(len(dataset), [0.7, 0.2, 0.1]), generator=generator)
    elif settings.case == "test":
        datasets = random_split(
            dataset, _get_splits(len(dataset), [0, 0, 1.0]), generator=generator)

    dataloaders = {}
    if settings.case in ["train", "finetune"]:
        dataloaders["train"] = DataLoader(datasets[0], batch_size=1000, shuffle=True, num_workers=8, pin_memory=True)
        dataloaders["val"] = DataLoader(datasets[1], batch_size=1000, shuffle=True, num_workers=8, pin_memory=True)
    dataloaders["test"] = DataLoader(datasets[2], batch_size=1000, shuffle=True, num_workers=8, pin_memory=True)

    return dataset, dataloaders


def run(settings: SettingsTraining):
    time_begin = time.perf_counter()
    timestamp_begin = time.ctime()

    dataset, dataloaders = init_data(settings)

    if settings.device is None:
        settings.device = "cuda" if cuda.is_available() else "cpu"
    logging.warning(f"Using {settings.device} device")

    # model choice
    in_channels = dataset.input_channels
    if settings.case in ["test", "finetune"]:
        model = load_model({"model_choice": settings.model_choice,
                           "in_channels": in_channels}, settings.path_to_model)
    else:
        model = create_model(settings.model_choice, in_channels)
    model.to(settings.device)

    # summary(model, (4, 256, 16))
    number_parameter = count_parameters(model)
    logging.warning(
        f"Model {settings.model_choice} with number of parameters: {number_parameter}")

    if settings.case in ["train", "finetune"]:
        # parameters of training
        loss_fn_str = "MSE"
        loss_fn = create_loss_fn(loss_fn_str, dataloaders)
        # training
        solver = Solver(model, dataloaders["train"], dataloaders["val"],
                        loss_func=loss_fn, finetune=settings.finetune)
        try:
            solver.load_lr_schedule(os.path.join(os.getcwd(), "runs", settings.name_folder_destination, "learning_rate_history.csv"))
            time_initializations = time.perf_counter()
            solver.train(settings)
            time_training = time.perf_counter()
        except KeyboardInterrupt:
            time_training = time.perf_counter()
            logging.warning("Stopping training early")
            logging.warning(f"Best model was found in epoch {solver.best_model_params['epoch']}.")
            compare_models(model, solver.best_model_params["state_dict"])
            
        finally:
            solver.save_lr_schedule(os.path.join(os.getcwd(), "runs", settings.name_folder_destination, "learning_rate_history.csv"))
    else:
        # load model (for application only)
        model = load_model({"model_choice": settings.model_choice, "in_channels": in_channels}, os.path.join(settings.path_to_model), "model", settings.device)
        model.to(settings.device)

    # save model
    save(model.state_dict(), os.path.join(os.getcwd(), "runs", settings.name_folder_destination, "model.pt"))

    # visualization
    try:
        avg_inference_times = []
        if settings.case in ["train", "finetune"]:
            plot_sample(model, dataloaders["val"], settings.device, plot_name=settings.name_folder_destination + "/plot_val_sample", amount_plots=10,)
            errors_val, avg_inference_time = error_measurements(model, dataloaders["val"], settings.device, plot_name=settings.name_folder_destination + "/plot_val")
            avg_inference_times.append(avg_inference_time)

            plot_sample(model, dataloaders["train"], settings.device, plot_name=settings.name_folder_destination + "/plot_train_sample", amount_plots=2,)
            errors_train, avg_inference_time = error_measurements(model, dataloaders["train"], settings.device, plot_name=settings.name_folder_destination + "/plot_train")
            avg_inference_times.append(avg_inference_time)
        else:
            plot_sample(model, dataloaders["test"], settings.device, plot_name=settings.name_folder_destination + "/plot_test_sample", amount_plots=10,)
            errors_test, avg_inference_time = error_measurements(model, dataloaders["test"], settings.device, plot_name=settings.name_folder_destination + "/plot_test")
            avg_inference_times.append(avg_inference_time)
    except:
        pass 

    time_end = time.perf_counter()
    duration = f"{(time_end-time_begin)//60} minutes {(time_end-time_begin)%60} seconds"
    print(f"Experiment took {duration}")

    # save measurements
    with open(os.path.join(os.getcwd(), "runs", settings.name_folder_destination, f"measurements_{settings.case}.yaml"), "w") as f:
        f.write(f"timestamp of beginning: {timestamp_begin}\n")
        f.write(f"timestamp of end: {time.ctime()}\n")
        f.write(f"duration of whole process including visualisation in seconds: {(time_end-time_begin)}\n")
        if settings.case in ["train", "finetune"]: 
            f.write(f"duration of initializations in seconds: {time_initializations-time_begin}\n")
            f.write(f"duration of training in seconds: {time_training-time_initializations}\n")
        f.write(f"model: {settings.model_choice}\n")
        f.write(f"number of input-channels: {in_channels}\n")
        f.write(f"input params: {settings.inputs_prep}\n")
        f.write(f"dataset location: {settings.datasets_path}\n")
        f.write(f"dataset name: {settings.dataset_name}\n")
        f.write(f"number of datapoints: {len(dataset)}\n")
        f.write(f"name_destination_folder: {settings.name_folder_destination}\n")
        f.write(f"number epochs: {settings.epochs}\n")
        try:
            if settings.case in ["train", "finetune"]: 
                f.write(f"errors train: {errors_train}\n")
                f.write(f"errors val: {errors_val}\n")
            else:
                f.write(f"errors test: {errors_test}\n")
            f.write(f"avg inference times in seconds: {avg_inference_times}\n")
        except:
            pass
        f.write(f"number parameters: {number_parameter}\n")
        f.write(f"device: {settings.device}\n")
        f.write(f"case: {settings.case}\n")
        if settings.case in ["test", "finetune"]: 
            f.write(f"path to pretrained model: {settings.path_to_model}\n")
        if settings.case in ["train", "finetune"]: 
            f.write(f"loss function: {loss_fn_str}\n")
            f.write(f"best model found after epoch: {solver.best_model_params['epoch']}\n")
            f.write(f"best model found with val loss: {solver.best_model_params['loss']}\n")
            f.write(f"best model found with train loss: {solver.best_model_params['train loss']}\n")
            f.write(f"best model found with val RMSE: {solver.best_model_params['val RMSE']}\n")
            f.write(f"best model found with train RMSE: {solver.best_model_params['train RMSE']}\n")
            f.write(f"best model found after training time in seconds: {solver.best_model_params['training time in sec']}\n")


def finetune_2HP_NN():
    logging.basicConfig(level=logging.WARNING)
    args = {}
    args["device"] = "cuda:3"
    args["epochs"] = 10000
    args["model_choice"] = "unet"
    args["case"] = "finetune"
    args["path_to_model"] = "current_unet_benchmark_dataset_2d_100datapoints_input_empty_T_0"
    args["datasets_path"] = "/home/pelzerja/pelzerja/test_nn/datasets_prepared/2HP_NN"
    args["inputs_prep"] = "gksio"
    args["dataset_name"] = "dataset_2hps_1fixed_100dp_2hp"
    args["name_folder_destination"] = "2HP_NN_finetune_100dp" #f"current_{settings.model_choice}_{settings.dataset_name}"

    settings = SettingsTraining(**args)

    destination_dir = pathlib.Path(os.getcwd(), "runs", settings.name_folder_destination)
    destination_dir.mkdir(parents=True, exist_ok=True)

    settings.save()
    run(settings)

if __name__ == "__main__":
    # level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    logging.basicConfig(level=logging.WARNING)
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="benchmark_dataset_2d_100datapoints")
    # benchmark_dataset_2d_20dp_2hps benchmark_testcases_4 benchmark_dataset_2d_100dp_vary_hp_loc benchmark_dataset_2d_100datapoints dataset3D_100dp_perm_vary dataset3D_100dp_perm_iso
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--epochs", type=int, default=25000)
    parser.add_argument("--case", type=str, default="train") # test finetune
    parser.add_argument("--path_to_model", type=str, default="benchmarkPLUSdataset_2d_100dp_vary_hp_loc/unet_inputs_pk_MatID_noPowerOf2") # for finetuning or testing
    parser.add_argument("--model_choice", type=str, default="unet")
    parser.add_argument("--name_folder_destination", type=str, default="")
    parser.add_argument("--inputs_prep", type=str, default="pksi")
    parser.add_argument("--name_extension", type=str, default="") # _grad_p
    args = parser.parse_args()
    
    default_raw_dir, datasets_prepared_dir, dataset_prepared_full_path = set_paths(args.dataset_name, args.inputs_prep, args.name_extension)
    args.datasets_path = datasets_prepared_dir

    # prepare dataset if not done yet
    if not os.path.exists(dataset_prepared_full_path):
        args_prep = {"raw_dir": default_raw_dir,
            "datasets_dir": datasets_prepared_dir,
            "dataset_name": args.dataset_name,
            "inputs_prep": args.inputs_prep,
            "name_extension": args.name_extension}
        args_prep = SettingsPrepare(**args_prep)
        prepare_dataset(args=args_prep)
        print(f"Dataset {dataset_prepared_full_path} prepared")

    else:
        print(f"Dataset {dataset_prepared_full_path} already prepared")
    args.dataset_name += "_"+args.inputs_prep + args.name_extension

    settings = SettingsTraining(**vars(args))
    if settings.name_folder_destination == "":
        settings.name_folder_destination = f"current_{settings.model_choice}_{settings.dataset_name}"
    destination_dir = pathlib.Path(os.getcwd(), "runs", settings.name_folder_destination)
    destination_dir.mkdir(parents=True, exist_ok=True)

    settings.save()
    run(settings)

    # beep()
    # tensorboard --logdir=runs/ --host localhost --port 8088
