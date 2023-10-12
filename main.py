import argparse
import logging
import os
import multiprocessing
import pathlib
import shutil
import time

import torch
from torch import save
from torch.utils.data import DataLoader, random_split
# tensorboard --logdir=runs/ --host localhost --port 8088
# from torch.utils.tensorboard import SummaryWriter
from torch.nn import MSELoss

from data_stuff.dataset import SimulationDataset, _get_splits
from data_stuff.utils import SettingsTraining
from networks.unet import UNet
from preprocessing.prepare_1ststage import prepare_dataset_for_1st_stage
from preprocessing.prepare_2ndstage import prepare_dataset_for_2nd_stage
from solver import Solver
from utils.prepare_paths import set_paths_1hpnn, Paths1HP, Paths2HP, set_paths_2hpnn
from utils.visualization import plt_avg_error_cellwise, plot_sample
from utils.measurements import measure_loss, save_all_measurements


def init_data(settings: SettingsTraining, seed=1):
    dataset = SimulationDataset(pathlib.Path(settings.datasets_dir) / settings.dataset_prep)
    generator = torch.Generator().manual_seed(seed)

    split_ratios = [0.7, 0.2, 0.1]
    # split_ratios = [0.0, 0.0, 1.0] 
    datasets = random_split(dataset, _get_splits(len(dataset), split_ratios), generator=generator)

    dataloaders = {}
    try:
        dataloaders["train"] = DataLoader(datasets[0], batch_size=1000, shuffle=True, num_workers=8)
        dataloaders["val"] = DataLoader(datasets[1], batch_size=1000, shuffle=True, num_workers=8)
    except: pass
    dataloaders["test"] = DataLoader(datasets[2], batch_size=1000, shuffle=True, num_workers=8)

    return dataset, dataloaders


def run(settings: SettingsTraining):
    multiprocessing.set_start_method("spawn", force=True)
    
    times = {}
    times["time_begin"] = time.perf_counter()
    times["timestamp_begin"] = time.ctime()

    dataset, dataloaders = init_data(settings)

    # model
    model = UNet(in_channels=dataset.input_channels).float()
    if settings.case in ["test", "finetune"]:
        model.load_state_dict(torch.load(f"{settings.model}/model.pt", map_location=torch.device(settings.device)))
    model.to(settings.device)

    solver = None
    if settings.case in ["train", "finetune"]:
        loss_fn = MSELoss()
        # training
        solver = Solver(model, dataloaders["train"], dataloaders["val"], loss_func=loss_fn, finetune=settings.finetune)
        try:
            solver.load_lr_schedule(settings.destination_dir / "learning_rate_history.csv", settings.case_2hp)
            times["time_initializations"] = time.perf_counter()
            solver.train(settings, settings.destination_dir)
            times["time_training"] = time.perf_counter()
        except KeyboardInterrupt:
            times["time_training"] = time.perf_counter()
            logging.warning(f"Manually stopping training early with best model found in epoch {solver.best_model_params['epoch']}.")
        finally:
            solver.save_lr_schedule(settings.destination_dir / "learning_rate_history.csv")

    # save model
    save(model.state_dict(), settings.destination_dir / "model.pt")

    # visualization
    if settings.visualize:
        plot_sample(model, dataloaders["val"], settings.device, plot_path=settings.destination_dir / "plot_val", amount_plots=5, pic_format="png")

    times["time_end"] = time.perf_counter()
    print(f"Experiment took {(times['time_end']-times['time_begin'])//60} minutes {(times['time_end']-times['time_begin'])%60} seconds")
    save_all_measurements(settings, len(dataset), times, solver) #, errors)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_raw", type=str, default="benchmark_dataset_2d_100datapoints", help="Name of the raw dataset (without inputs)")
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--case", type=str, default="train") # alternatives: test finetune
    parser.add_argument("--model", type=str, default="default") # required for testing or finetuning
    parser.add_argument("--destination_dir", type=str, default="")
    parser.add_argument("--inputs", type=str, default="gksi")
    parser.add_argument("--case_2hp", type=bool, default=False)
    parser.add_argument("--visualize", type=bool, default=False) # TODO
    settings = SettingsTraining(**vars(parser.parse_args()))

    if not settings.case_2hp:
        paths: Paths1HP
        paths, dataset_prep = set_paths_1hpnn(settings.dataset_raw, settings.inputs) 
        settings.datasets_dir = paths.datasets_prepared_dir
        settings.dataset_prep = dataset_prep

        # prepare dataset if not done yet OR if test=case do it anyways because of potentially different std,mean,... values than trained with
        if not os.path.exists(paths.dataset_1st_prep_path) or settings.case == "test":
            prepare_dataset_for_1st_stage(paths, settings)
        print(f"Dataset {paths.dataset_1st_prep_path} prepared")
    else:
        paths: Paths2HP
        paths, inputs_1hp, dataset_prep = set_paths_2hpnn(settings.dataset_raw, settings.inputs)
        settings.datasets_dir = paths.datasets_prepared_dir 
        settings.dataset_prep = dataset_prep

        if not os.path.exists(paths.datasets_boxes_prep_path):
            prepare_dataset_for_2nd_stage(paths, settings.dataset_raw, inputs_1hp, settings.device)
        print(f"Dataset {paths.datasets_boxes_prep_path} prepared")

    if settings.case == "train":
        shutil.copyfile(pathlib.Path(paths.datasets_prepared_dir) / paths.dataset_1st_prep_path / "info.yaml", settings.destination_dir / "info.yaml")
    settings.save()

    run(settings)