import argparse
import logging
from copy import deepcopy
import multiprocessing
import numpy as np
import time
import torch
from torch.utils.data import DataLoader, random_split
# tensorboard --logdir=runs/ --host localhost --port 8088
from torch.utils.tensorboard import SummaryWriter
from torch.nn import MSELoss

from preprocessing.data_stuff.dataset import SimulationDataset, SimulationDatasetCuts, _get_splits
from utils.utils_data import SettingsTraining
from processing.networks.unet import UNet
from processing.solver import Solver
from preprocessing.prepare import prepare_data_and_paths
from postprocessing.visualization import plot_avg_error_cellwise, visualizations, infer_all_and_summed_pic
from postprocessing.measurements import measure_loss, save_all_measurements

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# torch.cuda.empty_cache()

def init_data(settings: SettingsTraining, seed=1):
    if settings.case == "test":
        dataset = SimulationDataset(settings.dataset_prep)
    else:
        dataset = SimulationDatasetCuts(settings.dataset_prep, skip_per_dir=64)
    generator = torch.Generator().manual_seed(seed)

    split_ratios = [0.5, 0.3, 0.2]
    if settings.case == "test": # TODO change back
        split_ratios = [0.0, 0.0, 1.0] 
    datasets = random_split(dataset, _get_splits(len(dataset), split_ratios), generator=generator)

    dataloaders = {}
    try:
        dataloaders["train"] = DataLoader(datasets[0], batch_size=100, shuffle=True, num_workers=0)
        dataloaders["val"] = DataLoader(datasets[1], batch_size=100, shuffle=True, num_workers=0)
        print(len(datasets[0]), len(datasets[1]))
    except: pass
    dataloaders["test"] = DataLoader(datasets[2], batch_size=100, shuffle=True, num_workers=0)
    print(len(datasets[0]), len(datasets[1]), len(datasets[2]))
    return dataset.input_channels, dataloaders


def init_data_different_datasets(settings: SettingsTraining, settings_val: SettingsTraining = None, settings_test: SettingsTraining = None):
    dataloaders = {}

    if settings.case == "test":
        dataset = SimulationDataset(settings.dataset_prep)
        dataloaders["test"] = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=0)
    else:
        dataset = SimulationDatasetCuts(settings.dataset_prep, skip_per_dir=32)
        dataloaders["train"] = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=0)
        if settings_val:
            dataset_tmp = SimulationDatasetCuts(settings_val.dataset_prep, skip_per_dir=32)
            dataloaders["val"] = DataLoader(dataset_tmp, batch_size=100, shuffle=True, num_workers=0)
        if settings_test:
            dataset_tmp = SimulationDataset(settings_test.dataset_prep)
            dataloaders["test"] = DataLoader(dataset_tmp, batch_size=100, shuffle=True, num_workers=0)

        print(len(dataset), len(dataloaders["val"].dataset), len(dataloaders["test"].dataset))

    return dataset.input_channels, dataloaders


def run(settings: SettingsTraining, settings_val: SettingsTraining = None, settings_test: SettingsTraining = None):
    np.random.seed(1)
    multiprocessing.set_start_method("spawn", force=True)

    different_datasets = True
    
    times = {}
    times["time_begin"] = time.perf_counter()
    times["timestamp_begin"] = time.ctime()

    if different_datasets:
        input_channels, dataloaders = init_data_different_datasets(settings, settings_val, settings_test)
    else:
        input_channels, dataloaders = init_data(settings)

    # model
    model = UNet(in_channels=input_channels, depth=3).float()
    if settings.case in ["test", "finetune"]:
        model.load(settings.model, settings.device)
    model.to(settings.device)

    solver = None
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

    # save model
    model.save(settings.destination)

    # visualization
    which_dataset = "val"
    pic_format = "png"
    if settings.case == "test":
        settings.visualize = True
        which_dataset = "test"
    errors = measure_loss(model, dataloaders[which_dataset], settings.device)
    times["time_end"] = time.perf_counter()
    save_all_measurements(settings, len(dataloaders[which_dataset].dataset), times, solver, errors)

    if settings.visualize:
        if not different_datasets:
            visualizations(model, dataloaders[which_dataset], settings.device, plot_path=settings.destination / f"plot_{which_dataset}", amount_datapoints_to_visu=5, pic_format=pic_format, case=settings.case)
            times[f"avg_inference_time of {which_dataset}"], summed_error_pic = infer_all_and_summed_pic(model, dataloaders[which_dataset], settings.device)
            plot_avg_error_cellwise(dataloaders[which_dataset], summed_error_pic, {"folder" : settings.destination, "format": pic_format})
            print("Visualizations finished")
        else:
            # settings.device = "cpu"
            visualizations(model, dataloaders["val"], settings.device, plot_path=settings.destination / f"val", amount_datapoints_to_visu=1, pic_format=pic_format, case=settings.case)
            visualizations(model, dataloaders["test"], settings.device, plot_path=settings.destination / f"test", amount_datapoints_to_visu=1, pic_format=pic_format, case=settings.case)

            settings.case = "test"
            _, dataloaders = init_data(settings)
            visualizations(model, dataloaders["test"], settings.device, plot_path=settings.destination / f"train", amount_datapoints_to_visu=1, pic_format=pic_format, case=settings.case)
            settings.case = "train"
            print("Visualizations finished")

    print(f"Whole process took {(times['time_end']-times['time_begin'])//60} minutes {np.round((times['time_end']-times['time_begin'])%60, 1)} seconds\nOutput in {settings.destination.parent.name}/{settings.destination.name}")

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
    parser.add_argument("--dataset_train", type=str, default="dataset_giant_100hp_varyPermLog_p30_kfix_quarter_dp4_4", help="Name of the raw dataset (without inputs)")
    parser.add_argument("--dataset_val", type=str, default="dataset_giant_100hp_varyPermLog_p30_kfix_quarter_dp5_4", help="Name of the raw dataset (without inputs)")
    parser.add_argument("--dataset_test", type=str, default="dataset_giant_100hp_varyPermLog_p30_kfix_quarter_dp2_4", help="Name of the raw dataset (without inputs)")
    parser.add_argument("--dataset_prep", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--case", type=str, choices=["train", "test", "finetune"], default="train")
    parser.add_argument("--model", type=str, default="default") # required for testing or finetuning
    parser.add_argument("--destination", type=str, default="")
    parser.add_argument("--inputs", type=str, default="gksi") #choices=["gki", "gksi", "pksi", "gks", "gkmi", "lm", "lmi", "lmik","lmikp", "ls", "m", "t", "gkiab", "gksiab", "gkt", "gksi100", "ogksi1000", "gksi1000", "pksi100", "pksi1000", "ogksi1000_finetune", "gki100"]
    parser.add_argument("--visualize", type=bool, default=False)
    parser.add_argument("--save_inference", type=bool, default=False)
    parser.add_argument("--problem", type=str, choices=["extend", "allin1", "2stages"], default="allin1")
    parser.add_argument("--notes", type=str, default="")
    args = parser.parse_args()
    settings = SettingsTraining(**vars(args))

    settings = prepare_data_and_paths(settings, dataset_raw=settings.dataset_train)
    settings.case = "test"
    settings.model = settings.destination
    settings_val = prepare_data_and_paths(deepcopy(settings), dataset_raw=settings.dataset_val)
    settings_test = prepare_data_and_paths(deepcopy(settings), dataset_raw=settings.dataset_test)
    settings.case = "train"


    model = run(settings, settings_val, settings_test)

    if args.save_inference:
        save_inference(settings.model, len(args.inputs), settings)