import datetime as dt
import logging
import os
import argparse
from networks.models import create_model, load_model, compare_models
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
from torch.utils.tensorboard import SummaryWriter
from prepare_dataset import prepare_dataset

def init_data(settings: SettingsTraining, seed=1):
    dataset = SimulationDataset(
        os.path.join(settings.datasets_path, settings.dataset_name))
    generator = torch.Generator().manual_seed(seed)

    if settings.case in ["train", "finetune"]:
        datasets = random_split(
            dataset, _get_splits(len(dataset), [0.7, 0.2, 0.1]), generator=generator)
    elif settings.case == "test":
        datasets = random_split(
            dataset, _get_splits(len(dataset), [0, 0, 1.0]), generator=generator)

    dataloaders = {}
    if settings.case in ["train", "finetune"]:
        dataloaders["train"] = DataLoader(datasets[0], batch_size=1000, shuffle=True, num_workers=0)
        dataloaders["val"] = DataLoader(datasets[1], batch_size=1000, shuffle=True, num_workers=0)
    dataloaders["test"] = DataLoader(datasets[2], batch_size=1000, shuffle=True, num_workers=0)

    return dataset, dataloaders


def run(settings: SettingsTraining):
    time_begin = dt.datetime.now()

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
            solver.train(settings)
        except KeyboardInterrupt:
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
    if settings.case in ["train", "finetune"]:
        error_mean, final_max_error = plot_sample(model, dataloaders["val"], settings.device, plot_name=settings.name_folder_destination + "/plot_val_sample", amount_plots=10,)
        error_mean, final_max_error = plot_sample(model, dataloaders["train"], settings.device, plot_name=settings.name_folder_destination + "/plot_train_sample", amount_plots=2,)
    else:
        plot_sample(model, dataloaders["test"], settings.device, plot_name=settings.name_folder_destination + "/plot_test_sample", amount_plots=10,)

    time_end = dt.datetime.now()
    duration = f"{(time_end-time_begin).seconds//60} minutes {(time_end-time_begin).seconds%60} seconds"
    print(f"Experiment took {duration}")

    # logging
    if False:
        results = {"timestamp": time_begin, "model": settings.model_choice, "dataset": settings.dataset_name, "n_epochs": settings.epochs,
                "error_mean": error_mean[-1], "error_max": final_max_error, "duration": duration, "name_destination_folder": settings.name_folder_destination, }
        append_results_to_csv(results, "runs/collected_results_rough_idea.csv")

def _get_splits(n, splits):
    splits = [int(n * s) for s in splits[:-1]]
    splits.append(n - sum(splits))
    return splits

def set_paths(dataset_name: str, name_extension: str = None):
    # TODO reasonable defaults
    if os.path.exists("/scratch/sgs/pelzerja/"):
        default_raw_dir = "/scratch/sgs/pelzerja/datasets/1hp_boxes"
        datasets_prepared_dir="/home/pelzerja/pelzerja/test_nn/datasets_prepared/1HP_NN"
    else:
        default_raw_dir = "/home/pelzerja/Development/simulation_groundtruth_pflotran/Phd_simulation_groundtruth/datasets/1hp_boxes"
        datasets_prepared_dir = "/home/pelzerja/Development/datasets_prepared/1HP_NN"
    
    dataset_prepared_path = os.path.join(datasets_prepared_dir, dataset_name, name_extension)

    return default_raw_dir, datasets_prepared_dir, dataset_prepared_path

if __name__ == "__main__":
    # level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    logging.basicConfig(level=logging.WARNING)
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="benchmark_dataset_2d_100datapoints")
    # benchmark_dataset_2d_20dp_2hps benchmark_testcases_4 benchmark_dataset_2d_100dp_vary_hp_loc benchmark_dataset_2d_100datapoints dataset3D_100dp_perm_vary dataset3D_100dp_perm_iso
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--epochs", type=int, default=30000)
    parser.add_argument("--case", type=str, default="train") # test finetune
    parser.add_argument("--path_to_model", type=str, default="benchmarkPLUSdataset_2d_100dp_vary_hp_loc/unet_inputs_pk_MatID_noPowerOf2") # for finetuning or testing
    parser.add_argument("--model_choice", type=str, default="unet")
    parser.add_argument("--name_folder_destination", type=str, default="")
    parser.add_argument("--inputs_prep", type=str, default="pksi")
    args = parser.parse_args()
    
    name_extension = "_grad_p"
    default_raw_dir, datasets_prepared_dir, dataset_prepared_full_path = set_paths(args.dataset_name, name_extension)
    args.datasets_path = datasets_prepared_dir

    # prepare dataset if not done yet
    if not os.path.exists(dataset_prepared_full_path):
        prepare_dataset(raw_data_directory = default_raw_dir,
                        datasets_path = datasets_prepared_dir,
                        dataset_name = args.dataset_name,
                        input_variables = args.inputs_prep,
                        name_extension=name_extension,)

    else:
        print(f"Dataset {dataset_prepared_full_path} already prepared")
    args.dataset_name += name_extension

    settings = SettingsTraining(**vars(args))
    if settings.name_folder_destination == "":
        settings.name_folder_destination = f"current_{settings.model_choice}_{settings.dataset_name}"

    try:
        os.mkdir(os.path.join(os.getcwd(), "runs", settings.name_folder_destination))
    except FileExistsError:
        pass

    settings.save()
    run(settings)
    # tensorboard --logdir runs/
