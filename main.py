import argparse
import logging
import multiprocessing
import numpy as np
import shutil
import time

import torch
from torch.utils.data import DataLoader, random_split
# tensorboard --logdir=runs/ --host localhost --port 8088
from torch.utils.tensorboard import SummaryWriter
from torch.nn import MSELoss

from data_stuff.dataset import SimulationDataset, _get_splits
from data_stuff.utils import SettingsTraining
from networks.unet import UNet
from preprocessing.prepare_1ststage import prepare_dataset_for_1st_stage
from preprocessing.prepare_2ndstage import prepare_dataset_for_2nd_stage
from solvers.solver import Solver
from utils.prepare_paths import set_paths_1hpnn, Paths1HP, Paths2HP, set_paths_2hpnn
from utils.visualization import plot_avg_error_cellwise, visualizations, infer_all_and_summed_pic
from utils.measurements import measure_loss, save_all_measurements


def init_data(settings: SettingsTraining, seed=1):
    dataset = SimulationDataset(settings.dataset_prep)
    generator = torch.Generator().manual_seed(seed)

    split_ratios = [0.7, 0.2, 0.1]
    # split_ratios = [0.0, 0.0, 1.0] 
    datasets = random_split(dataset, _get_splits(len(dataset), split_ratios), generator=generator)

    dataloaders = {}
    try:
        dataloaders["train"] = DataLoader(datasets[0], batch_size=1000, shuffle=True, num_workers=0)
        dataloaders["val"] = DataLoader(datasets[1], batch_size=1000, shuffle=True, num_workers=0)
    except: pass
    dataloaders["test"] = DataLoader(datasets[2], batch_size=1000, shuffle=True, num_workers=0)

    return dataset, dataloaders


def prepare_data_and_paths(settings:SettingsTraining):
    if not settings.case_2hp:
        paths: Paths1HP
        paths, destination_dir = set_paths_1hpnn(settings.dataset_raw, settings.inputs) 
        settings.dataset_prep = paths.dataset_1st_prep_path

    else:
        paths: Paths2HP
        paths, inputs_1hp, destination_dir = set_paths_2hpnn(settings.dataset_raw, settings.inputs)
        settings.dataset_prep = paths.datasets_boxes_prep_path
    settings.make_destination_path(destination_dir)
    settings.make_model_path(destination_dir)

    if not settings.case_2hp:
        # prepare dataset if not done yet OR if test=case do it anyways because of potentially different std,mean,... values than trained with
        if not paths.dataset_1st_prep_path.exists(): # or settings.case == "test":
            prepare_dataset_for_1st_stage(paths, settings)
        print(f"Dataset prepared ({paths.dataset_1st_prep_path})")

    else:
        if not paths.datasets_boxes_prep_path.exists() or not paths.dataset_1st_prep_path: # TODO settings.case == "test"?
            prepare_dataset_for_2nd_stage(paths, inputs_1hp, settings.device)
        print(f"Dataset prepared ({paths.datasets_boxes_prep_path})")

    if settings.case == "train":
        shutil.copyfile(paths.dataset_1st_prep_path / "info.yaml", settings.destination / "info.yaml")
    settings.save()
    return settings


def run(settings: SettingsTraining):
    multiprocessing.set_start_method("spawn", force=True)
    
    times = {}
    times["time_begin"] = time.perf_counter()
    times["timestamp_begin"] = time.ctime()

    dataset, dataloaders = init_data(settings)

    # model
    model = UNet(in_channels=dataset.input_channels).float()
    if settings.case in ["test", "finetune"]:
        model.load(settings.model, settings.device)
    model.to(settings.device)

    solver = None
    if settings.case in ["train", "finetune"]:
        loss_fn = MSELoss()
        # training
        solver = Solver(model, dataloaders["train"], dataloaders["val"], loss_func=loss_fn, finetune=settings.finetune)
        try:
            solver.load_lr_schedule(settings.destination / "learning_rate_history.csv", settings.case_2hp)
            times["time_initializations"] = time.perf_counter()
            solver.train(settings)
            times["time_training"] = time.perf_counter()
        except KeyboardInterrupt:
            times["time_training"] = time.perf_counter()
            logging.warning(f"Manually stopping training early with best model found in epoch {solver.best_model_params['epoch']}.")
        finally:
            solver.save_lr_schedule(settings.destination / "learning_rate_history.csv")
            print("Training finished")

    # save model
    model.save(settings.destination)

    # visualization
    if settings.visualize:
        which_dataset = "val"
        pic_format = "png"
        visualizations(model, dataloaders[which_dataset], settings.device, plot_path=settings.destination / f"plot_{which_dataset}", amount_datapoints_to_visu=1, pic_format=pic_format)
        times[f"avg_inference_time of {which_dataset}"], summed_error_pic = infer_all_and_summed_pic(model, dataloaders[which_dataset], settings.device)
        plot_avg_error_cellwise(dataloaders[which_dataset], summed_error_pic, {"folder" : settings.destination, "format": pic_format})
        errors = measure_loss(model, dataloaders[which_dataset], settings.device)
        print("Visualizations finished")
        
    times["time_end"] = time.perf_counter()
    save_all_measurements(settings, len(dataset), times, solver) #, errors)
    print(f"Whole process took {(times['time_end']-times['time_begin'])//60} minutes {np.round((times['time_end']-times['time_begin'])%60, 1)} seconds\nOutput in {settings.destination.parent.name}/{settings.destination.name}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_raw", type=str, default="dataset_2d_small_1000dp", help="Name of the raw dataset (without inputs)")
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--case", type=str, default="train") # alternatives: test finetune
    parser.add_argument("--model", type=str, default="default") # required for testing or finetuning
    parser.add_argument("--destination", type=str, default="")
    parser.add_argument("--inputs", type=str, default="gksi")
    parser.add_argument("--case_2hp", type=bool, default=False)
    parser.add_argument("--visualize", type=bool, default=False)
    settings = SettingsTraining(**vars(parser.parse_args()))

    settings = prepare_data_and_paths(settings)

    run(settings)