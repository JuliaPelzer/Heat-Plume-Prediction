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
# tensorboard --logdir=runs/ --host localhost --port 8088
from torch.utils.tensorboard import SummaryWriter
import multiprocessing
from torch.nn import MSELoss

from data_stuff.dataset import SimulationDataset, _get_splits
from data_stuff.utils import SettingsTraining, SettingsPrepare
from networks.unet import UNet
from prepare_dataset import pre_prepare_dataset
from solver import Solver
from utils.utils import set_paths
from utils.visualize_data import plt_avg_error_cellwise, plot_sample, infer_all_and_summed_pic
from utils.measurements import measure_loss, save_all_measurements, measure_additional_losses

def init_data(settings: SettingsTraining, seed=1):
    dataset = SimulationDataset(os.path.join(settings.datasets_path, settings.dataset_name))
    generator = torch.Generator().manual_seed(seed)

    split_ratios = [0.7, 0.2, 0.1]
    # if settings.case == "test": # TODO change back
        # split_ratios = [0.0, 0.0, 1.0] 
    datasets = random_split(dataset, _get_splits(len(dataset), split_ratios), generator=generator)

    dataloaders = {}
    try:
        dataloaders["train"] = DataLoader(datasets[0], batch_size=1000, shuffle=True, num_workers=0)
        dataloaders["val"] = DataLoader(datasets[1], batch_size=1000, shuffle=True, num_workers=0)
    except: pass
    dataloaders["test"] = DataLoader(datasets[2], batch_size=1000, shuffle=True, num_workers=0)

    return dataset, dataloaders


def run(settings: SettingsTraining):
    multiprocessing.set_start_method("spawn", force=True)
    
    times = {}
    times["time_begin"] = time.perf_counter()
    times["timestamp_begin"] = time.ctime()

    dataset, dataloaders = init_data(settings)

    # model
    model = UNet(in_channels=dataset.input_channels, out_channels=1, depth=3, kernel_size=5).float()
    if settings.case in ["test", "finetune"]:
        model.load_state_dict(torch.load(f"/scratch/sgs/pelzerja/models/paper23/best_models_2hpnn/{settings.path_to_model}/model.pt", map_location=torch.device(settings.device)))
    model.to(settings.device)

    solver = None
    if settings.case in ["train", "finetune"]:
        loss_fn = MSELoss()
        # training
        solver = Solver(model, dataloaders["train"], dataloaders["val"], loss_func=loss_fn, finetune=settings.finetune)
        try:
            solver.load_lr_schedule(os.path.join(os.getcwd(), "runs", settings.name_folder_destination, "learning_rate_history.csv"), settings.case_2hp)
            times["time_initializations"] = time.perf_counter()
            solver.train(settings, settings.name_folder_destination)
            times["time_training"] = time.perf_counter()
        except KeyboardInterrupt:
            times["time_training"] = time.perf_counter()
            logging.warning(f"Manually stopping training early with best model found in epoch {solver.best_model_params['epoch']}.")
        finally:
            solver.save_lr_schedule(os.path.join(os.getcwd(), "runs", settings.name_folder_destination, "learning_rate_history.csv"))

    # save model
    save(model.state_dict(), os.path.join(os.getcwd(), "runs", settings.name_folder_destination, "model.pt"))

    # visualization
    which_dataset = "val"
    pic_format = "png"
    if settings.case == "test":
        settings.visualize = True
    for which_dataset in ["test", "train", "val"]:
        # visualizations(model, dataloaders[which_dataset], settings.device, plot_path=settings.destination / f"plot_{which_dataset}", amount_datapoints_to_visu=5, pic_format=pic_format)
        times[f"avg_inference_time of {which_dataset}"], summed_error_pic = infer_all_and_summed_pic(model, dataloaders[which_dataset], settings.device)
        # plot_avg_error_cellwise(dataloaders[which_dataset], summed_error_pic, {"folder" : settings.destination, "format": pic_format})
        errors = measure_loss(model, dataloaders[which_dataset], settings.device)
        print("Visualizations finished")
        times["time_end"] = time.perf_counter()
        save_all_measurements(settings, len(dataset), times, solver, errors, which_dataset)
    measure_additional_losses(model, dataloaders, settings.device, summed_error_pic, settings)
        
    # print(f"Whole process took {(times['time_end']-times['time_begin'])//60} minutes {np.round((times['time_end']-times['time_begin'])%60, 1)} seconds\nOutput in {settings.destination.parent.name}/{settings.destination.name}")

    return model

def save_inference(model_name:str, in_channels: int, settings: SettingsTraining):
    # push all datapoints through and save all outputs
    model = UNet(in_channels=in_channels).float()
    model.load(model_name, settings.device)
    model.eval()

    data_dir = settings.dataset_prep
    (data_dir / "Outputs").mkdir(exist_ok=True)

    for datapoint in (data_dir / "Inputs").iterdir():
        data = torch.load(datapoint)
        data = torch.unsqueeze(data, 0)
        y_out = model(data.to(settings.device)).to(settings.device)
        y_out = y_out.detach().cpu()
        y_out = torch.squeeze(y_out, 0)
        torch.save(y_out, data_dir / "Outputs" / datapoint.name)
    
    print(f"Inference finished, outputs saved in {data_dir / 'Outputs'}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="benchmark_dataset_2d_100datapoints")
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--case", type=str, default="train") # test finetune
    parser.add_argument("--path_to_model", type=str, default="default") #for finetuning or testing
    parser.add_argument("--name_folder_destination", type=str, default="")
    parser.add_argument("--inputs_prep", type=str, default="gksi")
    parser.add_argument("--case_2hp", type=bool, default=False)
    args = parser.parse_args()
    
    default_raw_dir, args.datasets_path, dataset_prepared_full_path = set_paths(args.dataset_name, args.inputs_prep, args.case_2hp)

    # prepare dataset if not done yet OR if test=case do it anyways because of potentially different std,mean,... values than trained with
    if not os.path.exists(dataset_prepared_full_path) or (args.case == "test" and not args.case_2hp):
        pre_prepare_dataset(args, default_raw_dir, dataset_prepared_full_path)
    print(f"Dataset {dataset_prepared_full_path} prepared")

    settings = SettingsTraining(**vars(args))
    run(settings)

    # tensorboard --logdir=runs/ --host localhost --port 8088
