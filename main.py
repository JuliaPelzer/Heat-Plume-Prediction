import argparse
import logging
import multiprocessing
import time
from copy import deepcopy

import numpy as np
import torch
import yaml
# tensorboard --logdir=runs/ --host localhost --port 8088
from torch.nn import MSELoss
from torch.utils.data import DataLoader, random_split

from postprocessing.measurements import (measure_additional_losses,
                                         measure_loss, save_all_measurements)
from postprocessing.visualization import (infer_all_and_summed_pic, plot_avg_error_cellwise, visualizations)
from preprocessing.datasets.dataset import get_splits
from preprocessing.datasets.dataset_1stbox import Dataset1stBox
from preprocessing.datasets.dataset_cuts_jit import SimulationDatasetCuts
from preprocessing.datasets.dataset_extend import DatasetExtend, DatasetEncoder, random_split_extend
from preprocessing.prepare_overview import prepare_data_and_paths
from preprocessing.prepare_allin1 import preprocessing_allin1
from processing.networks.encoder import Encoder
from processing.networks.unet import UNet
from processing.networks.unetVariants import UNetHalfPad, UNetHalfPad2
from processing.solver import Solver
from utils.utils_data import SettingsTraining

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# torch.cuda.empty_cache()

def init_data(settings: SettingsTraining, seed=1):
    if settings.problem == "2stages":
        dataset = Dataset1stBox(settings.dataset_prep, box_size=settings.len_box)
    elif settings.problem == "extend":
        dataset = DatasetExtend(settings.dataset_prep, box_size=settings.len_box, skip_per_dir=settings.skip_per_dir)
        # dataset = DatasetEncoder(settings.dataset_prep, box_size=settings.len_box, skip_per_dir=settings.skip_per_dir)
        settings.inputs += "T"
    elif settings.problem == "allin1":
        if settings.case == "test":
            dataset = Dataset1stBox(settings.dataset_prep)
        else:
            dataset = SimulationDatasetCuts(settings.dataset_prep, skip_per_dir=64)
    print(f"Length of dataset: {len(dataset)}")
    generator = torch.Generator().manual_seed(seed)

    split_ratios = [0.7, 0.2, 0.1]
    # if settings.case == "test": # TODO change back
    #     split_ratios = [0.0, 0.0, 1.0] 
    if not settings.problem == "extend":
        datasets = random_split(dataset, get_splits(len(dataset), split_ratios), generator=generator)
    else:
        datasets = random_split_extend(dataset, get_splits(len(dataset.input_names), split_ratios), generator=generator)

    dataloaders = {}
    try:
        dataloaders["train"] = DataLoader(datasets[0], batch_size=100, shuffle=True, num_workers=0)
        dataloaders["val"] = DataLoader(datasets[1], batch_size=100, shuffle=True, num_workers=0)
    except: pass
    dataloaders["test"] = DataLoader(datasets[2], batch_size=100, shuffle=True, num_workers=0)
    print(len(datasets[0]), len(datasets[1]), len(datasets[2]))

    return dataset.input_channels, dataloaders

def init_data_different_datasets(settings: SettingsTraining, settings_val: SettingsTraining = None, settings_test: SettingsTraining = None):
    dataloaders = {}

    if settings.case == "test":
        dataset = Dataset1stBox(settings.dataset_prep)
        dataloaders["test"] = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=0)
    else:
        dataset = SimulationDatasetCuts(settings.dataset_prep, skip_per_dir=settings.skip_per_dir, box_size=settings.len_box)
        dataloaders["train"] = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=0)
        if settings_val:
            dataset_tmp = SimulationDatasetCuts(settings_val.dataset_prep, skip_per_dir=settings.skip_per_dir, box_size=settings.len_box)
            dataloaders["val"] = DataLoader(dataset_tmp, batch_size=100, shuffle=True, num_workers=0)
        if settings_test:
            dataset_tmp = Dataset1stBox(settings_test.dataset_prep)
            dataloaders["test"] = DataLoader(dataset_tmp, batch_size=100, shuffle=True, num_workers=0)

        print(len(dataset), len(dataloaders["val"].dataset), len(dataloaders["test"].dataset))
    return dataset.input_channels, dataloaders


def run(settings: SettingsTraining, settings_val: SettingsTraining = None, settings_test: SettingsTraining = None, different_datasets: bool = False):
    np.random.seed(1)
    multiprocessing.set_start_method("spawn", force=True)

    
    times = {}
    times["time_begin"] = time.perf_counter()
    times["timestamp_begin"] = time.ctime()

    if different_datasets:
        input_channels, dataloaders = init_data_different_datasets(settings, settings_val, settings_test)
    else:
        input_channels, dataloaders = init_data(settings)

    # model
    if settings.problem in ["2stages", "allin1"]:
        model = UNet(in_channels=input_channels).float()
    elif settings.problem == "extend":
        model = UNetHalfPad2(in_channels=input_channels).float()
        # model = Encoder(in_channels=input_channels).float()

    if settings.case in ["test", "finetune"]:
        model.load(settings.model, settings.device)
    model.to(settings.device)

    if settings.case in ["train", "finetune"]:
        loss_fn = MSELoss()
        # training
        finetune = True if settings.case == "finetune" else False
        solver = Solver(model, dataloaders["train"], dataloaders["val"], loss_func=loss_fn, finetune=finetune)
        try:
            solver.load_lr_schedule(settings.destination / "learning_rate_history.csv")
            times["time_initializations"] = time.perf_counter()
            solver.train(settings)
            times["time_training"] = time.perf_counter()
        except KeyboardInterrupt:
            times["time_training"] = time.perf_counter()
            logging.warning(f"Manually stopping training early with best model found in epoch {solver.best_model_params['epoch']}.")
        finally:
            solver.save_lr_schedule(settings.destination / "learning_rate_history.csv")
            print("Training finished")
    else:
        solver = None

    # save model
    model.save(settings.destination)

    # visualization
    which_dataset = "val"
    pic_format = "png"
    times["time_end"] = time.perf_counter()
    if settings.case == "test":
        settings.visualize = True
        which_dataset = "test"
        # errors = measure_loss(model, dataloaders[which_dataset], settings.device)
    save_all_measurements(settings, len(dataloaders[which_dataset].dataset), times, solver) #, errors)
    if settings.visualize:
        if not different_datasets:
            visualizations(model, dataloaders[which_dataset], settings, plot_path=settings.destination / f"plot_{which_dataset}", amount_datapoints_to_visu=5, pic_format=pic_format, different_datasets=different_datasets)
            # times[f"avg_inference_time of {which_dataset}"], summed_error_pic = infer_all_and_summed_pic(model, dataloaders[which_dataset], settings.device)
            # plot_avg_error_cellwise(dataloaders[which_dataset], summed_error_pic, {"folder" : settings.destination, "format": pic_format})
            print("Visualizations finished")
        else:
            # settings.device = "cpu"
            case_tmp = settings.case
            try:
                visualizations(model, dataloaders["val"], settings, plot_path=settings.destination / f"val", amount_datapoints_to_visu=1, pic_format=pic_format, different_datasets=different_datasets)
            except: pass
            visualizations(model, dataloaders["test"], settings, plot_path=settings.destination / f"test", amount_datapoints_to_visu=1, pic_format=pic_format, different_datasets=different_datasets)

            settings.case = "test"
            _, dataloaders = init_data(settings)
            visualizations(model, dataloaders["test"], settings, plot_path=settings.destination / f"train", amount_datapoints_to_visu=1, pic_format=pic_format)
            settings.case = case_tmp
            print("Visualizations finished")
    # measure_additional_losses(model, dataloaders, settings.device, summed_error_pic, settings)

    print(f"Whole process took {(times['time_end']-times['time_begin'])//60} minutes {np.round((times['time_end']-times['time_begin'])%60, 1)} seconds\nOutput in {settings.destination.parent.name}/{settings.destination.name}")

    return model

def save_inference(model_name:str, in_channels: int, settings: SettingsTraining):
    # push all datapoints through and save all outputs
    if settings.problem == "2stages":
        model = UNet(in_channels=in_channels).float()
    elif settings.problem == "extend":
        model = UNetHalfPad(in_channels=in_channels).float()
    model.load(model_name, settings.device)
    model.eval()

    data_dir = settings.dataset_prep
    (data_dir / "Outputs").mkdir(exist_ok=True)

    for datapoint in (data_dir / "Inputs").iterdir():
        data = torch.load(datapoint)
        data = torch.unsqueeze(data, 0)
        time_start = time.perf_counter()
        y_out = model(data.to(settings.device)).to(settings.device)
        time_end = time.perf_counter()
        y_out = y_out.detach().cpu()
        y_out = torch.squeeze(y_out, 0)
        torch.save(y_out, data_dir / "Outputs" / datapoint.name)
        print(f"Inference of {datapoint.name} took {time_end-time_start} seconds")
    
    print(f"Inference finished, outputs saved in {data_dir / 'Outputs'}")

def main(args):
    try:
        settings = SettingsTraining(**vars(args))
    except:
        settings = SettingsTraining(**args)

    different_datasets = True
    if settings.problem == "allin1" and different_datasets:
        case_tmp = settings.case
        settings.dataset_raw = settings.dataset_train
        dataset_tmp = settings.dataset_raw
        # settings = prepare_data_and_paths(settings)
        settings = preprocessing_allin1(settings)
        prep_tmp = settings.dataset_prep
        settings.dataset_prep = ""
        settings.case = "test"
        settings.model = settings.destination
        settings.dataset_raw = settings.dataset_val
        # settings_val = prepare_data_and_paths(deepcopy(settings))
        settings_val = preprocessing_allin1(deepcopy(settings))
        settings.dataset_raw = settings.dataset_test
        # settings_test = prepare_data_and_paths(deepcopy(settings))
        settings_test = preprocessing_allin1(deepcopy(settings))
        settings.case = case_tmp
        settings.dataset_raw = dataset_tmp
        settings.dataset_prep = prep_tmp
        
        model = run(settings, settings_val, settings_test, different_datasets=different_datasets)
    elif settings.problem == "allin1":
        settings.dataset_raw = settings.dataset_train
        settings = prepare_data_and_paths(settings)
        model = run(settings)
    else:    
        settings = prepare_data_and_paths(settings)
        model = run(settings)

    if settings.save_inference:
        save_inference(settings.model, len(args.inputs), settings)

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_raw", type=str, default="dataset_2d_small_1000dp", help="Name of the raw dataset (without inputs)") # for problem: 2stages, extend
    parser.add_argument("--dataset_train", type=str, default="dataset_giant_100hp_varyPermLog_p30_kfix_quarter_dp4_4", help="Name of the raw dataset (without inputs)") # for problem: allin1
    parser.add_argument("--dataset_val", type=str, default="dataset_giant_100hp_varyPermLog_p30_kfix_quarter_dp5_4", help="Name of the raw dataset (without inputs)") # for problem: allin1
    parser.add_argument("--dataset_test", type=str, default="dataset_giant_100hp_varyPermLog_p30_kfix_quarter_dp3_4", help="Name of the raw dataset (without inputs)") #for problem: allin1
    parser.add_argument("--dataset_prep", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--case", type=str, choices=["train", "test", "finetune"], default="train")
    parser.add_argument("--model", type=str, default="default") # required for testing or finetuning
    parser.add_argument("--destination", type=str, default="")
    parser.add_argument("--inputs", type=str, default="gksi") #e.g. "gki", "gksi100", "ogksi1000_finetune", "t", "lmi", "lmik","lmikp", ...
    parser.add_argument("--case_2hp", type=bool, default=False)
    parser.add_argument("--visualize", type=bool, default=False)
    parser.add_argument("--save_inference", type=bool, default=False)
    parser.add_argument("--problem", type=str, choices=["2stages", "allin1",  "extend",], default="allin1")
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--len_box", type=int, default=64) # for extend:256, for 2stages=None
    parser.add_argument("--skip_per_dir", type=int, default=32)
    args = parser.parse_args()

    main(args)