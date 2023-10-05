import argparse
import datetime as dt
import logging
import os
import pathlib
import time
import yaml

import torch
from torch import save
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.nn import MSELoss

from data_stuff.dataset import SimulationDataset, _get_splits
from data_stuff.utils import SettingsTraining, SettingsPrepare
from networks.unet import UNet
from preprocessing.prepare_1ststage import prepare_dataset_for_1st_stage
from solver import Solver
from utils.utils import set_paths
from utils.visualization import plt_avg_error_cellwise, plot_sample
from utils.measurements import measure_loss, save_all_measurements

def init_data(settings: SettingsTraining, seed=1):
    dataset = SimulationDataset(os.path.join(settings.datasets_folder, settings.dataset))
    generator = torch.Generator().manual_seed(seed)

    split_ratios = [0.7, 0.2, 0.1]
    # split_ratios = [0.0, 0.0, 1.0] 
    datasets = random_split(dataset, _get_splits(len(dataset), split_ratios), generator=generator)

    dataloaders = {}
    try:
        dataloaders["train"] = DataLoader(datasets[0], batch_size=1000, shuffle=True, num_workers=8, pin_memory=True)
        dataloaders["val"] = DataLoader(datasets[1], batch_size=1000, shuffle=True, num_workers=8, pin_memory=True)
    except: pass
    dataloaders["test"] = DataLoader(datasets[2], batch_size=1000, shuffle=True, num_workers=8, pin_memory=True)

    return dataset, dataloaders


def run(settings: SettingsTraining):
    times = {}
    times["time_begin"] = time.perf_counter()
    times["timestamp_begin"] = time.ctime()
    logging.warning(f"On {settings.device} device")

    dataset, dataloaders = init_data(settings)

    # model
    model = UNet(in_channels=dataset.input_channels).float()
    if settings.case in ["test", "finetune"]:
        model.load_state_dict(torch.load(f"{settings.model_path}/model.pt", map_location=torch.device(settings.device)))
    model.to(settings.device)

    if settings.case in ["train", "finetune"]:
        loss_fn = MSELoss()
        # training
        solver = Solver(model, dataloaders["train"], dataloaders["val"], loss_func=loss_fn, finetune=settings.finetune)
        try:
            solver.load_lr_schedule(os.path.join(os.getcwd(), "runs", settings.destination_folder, "learning_rate_history.csv"), settings.case_2hp)
            times["time_initializations"] = time.perf_counter()
            solver.train(settings, settings.destination_folder)
            times["time_training"] = time.perf_counter()
        except KeyboardInterrupt:
            times["time_training"] = time.perf_counter()
            logging.warning(f"Manually stopping training early with best model found in epoch {solver.best_model_params['epoch']}.")
        finally:
            solver.save_lr_schedule(os.path.join(os.getcwd(), "runs", settings.destination_folder, "learning_rate_history.csv"))

    # save model
    save(model.state_dict(), os.path.join(os.getcwd(), "runs", settings.destination_folder, "model.pt"))

    # visualization
    if settings.visualize:
        plot_sample(model, dataloaders["val"], settings.device, plot_name=settings.destination_folder + "/plot_test_sample", amount_plots=5,)
        # times["avg_inference_time"] = {"test", plt_avg_error_cellwise(model, dataloaders["test"], settings.device, plot_name=settings.name_folder_destination + "/plot_test")}
        # errors = {}
        # errors["errors_test"] = measure_loss(model, dataloaders["test"], settings.device)
        # try:
        #     errors["errors_train"] = measure_loss(model, dataloaders["train"], settings.device)
        #     errors["errors_val"] = measure_loss(model, dataloaders["val"], settings.device)
        # except: pass

    times["time_end"] = time.perf_counter()
    print(f"Experiment took {(times['time_end']-times['time_begin'])//60} minutes {(times['time_end']-times['time_begin'])%60} seconds")
    save_all_measurements(settings, len(dataset), solver, times) #, errors)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="benchmark_dataset_2d_100datapoints")
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--case", type=str, default="train") # test finetune
    parser.add_argument("--model_path", type=str, default="default") #for finetuning or testing
    parser.add_argument("--destination_folder", type=str, default="")
    parser.add_argument("--inputs", type=str, default="gksi")
    parser.add_argument("--case_2hp", type=bool, default=False)
    parser.add_argument("--visualize", type=bool, default=False)
    args = parser.parse_args()
    
    default_raw_dir, args.datasets_folder, dataset_prepared_full_path = set_paths(args.dataset, args.inputs, args.case_2hp)

    # prepare dataset if not done yet OR if test=case do it anyways because of potentially different std,mean,... values than trained with
    if not os.path.exists(dataset_prepared_full_path) or (args.case == "test" and not args.case_2hp):
        prepare_dataset_for_1st_stage(args, default_raw_dir, dataset_prepared_full_path)
    print(f"Dataset {dataset_prepared_full_path} prepared")

    settings = SettingsTraining(**vars(args))
    run(settings)

    # tensorboard --logdir=runs/ --host localhost --port 8088
