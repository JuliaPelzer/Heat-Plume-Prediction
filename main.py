import argparse
import logging
import multiprocessing
import numpy as np
import time
import torch
import yaml
from torch.utils.data import DataLoader, random_split
from torch.nn import MSELoss

from data_stuff.dataset import SimulationDataset, get_splits
from data_stuff.utils import SettingsTraining
from networks.unet import UNet
from networks.unetQuad import UNetQuad
from networks.unetParallel import UNetParallel
from processing.solver import Solver
from processing.finetune import tune_nn
from preprocessing.prepare import prepare_data_and_paths
from postprocessing.visualization import plot_avg_error_cellwise, visualizations, infer_all_and_summed_pic, visualize_dataset
from postprocessing.measurements import measure_loss, save_all_measurements
from postprocessing.test_temp import test_temp
from torchsummary import summary

def init_data(settings: SettingsTraining, seed=1):
    dataset = SimulationDataset(settings.dataset_prep)
    print(f"Length of dataset: {len(dataset)}")
    generator = torch.Generator().manual_seed(seed)

    split_ratios = [0.7, 0.2, 0.1]
    if settings.case in ["test","visualize"]:
        split_ratios = [0.0, 0.0, 1.0]

    datasets = random_split(dataset, get_splits(len(dataset), split_ratios), generator=generator)
    dataloaders = {}
    try:
        dataloaders["train"] = DataLoader(datasets[0], batch_size=40, shuffle=True, num_workers=0)
        dataloaders["val"] = DataLoader(datasets[1], batch_size=40, shuffle=True, num_workers=0)
    except: pass
    dataloaders["test"] = DataLoader(datasets[2], batch_size=10, shuffle=True, num_workers=0)

    return dataset.input_channels, dataloaders


def run(settings: SettingsTraining):
    multiprocessing.set_start_method("spawn", force=True)

    if settings.case in ["hypertune"]:
        tune_nn(settings)
        return
    
    times = {}
    times["time_begin"] = time.perf_counter()
    times["timestamp_begin"] = time.ctime()

    input_channels, dataloaders = init_data(settings)
    # model
    if settings.problem == "2stages":
        model = UNet(in_channels=input_channels).float()
    elif settings.problem == "parallel":
        model = UNetParallel(in_channels=input_channels).float()
    elif settings.problem == "quad":
        model = UNetQuad(in_channels=input_channels).float()

    if settings.case in ["test", "finetune"]:
        model.load(settings.model, map_location=settings.device)
    
    if settings.case == "visualize":
        visualize_dataset(dataloaders["test"], settings.device, plot_path=settings.destination / f"plot_vis", amount_datapoints_to_visu=20, pic_format="png")
        return

    if settings.case == "iterative":
        model.to("cpu")
        model = test_temp(model,settings)
        return model
    
    model.to(settings.device)
    solver = None
    if settings.case in ["train", "finetune"]:
        loss_fn = MSELoss()
        # training
        finetune = True if settings.case == "finetune" else False
        solver = Solver(model, dataloaders["train"], dataloaders["val"], loss_func=loss_fn, finetune=finetune)
        try:
            solver.load_lr_schedule(settings.destination / "learning_rate_history.csv", False)
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
    summary(model,input_size=(5,64,256))

    # visualization
    which_dataset = "val"
    pic_format = "png"
    times["time_end"] = time.perf_counter()
    errors = {}
    if settings.case == "test":
        #summary(model, (5, 256, 64), receptive_field=True, max_depth=5)
        settings.visualize = True
        which_dataset = "test"
        errors = measure_loss(model, dataloaders, settings.device, vT_case="temperature")
        # errors = measure_loss(model, dataloaders[which_dataset], settings.device)
    if settings.visualize:
        errors["isolines"] = visualizations(model, dataloaders[which_dataset], settings.device, plot_path=settings.destination / f"plot_{which_dataset}", amount_datapoints_to_visu=10, pic_format=pic_format)
        times[f"avg_inference_time of {which_dataset}"], summed_error_pic = infer_all_and_summed_pic(model, dataloaders[which_dataset], settings.device)
        plot_avg_error_cellwise(dataloaders[which_dataset], summed_error_pic, {"folder" : settings.destination, "format": pic_format})
        print("Visualizations finished")
    save_all_measurements(settings, len(dataloaders[which_dataset].dataset), times, solver, errors)       
    print(f"Whole process took {(times['time_end']-times['time_begin'])//60} minutes {np.round((times['time_end']-times['time_begin'])%60, 1)} seconds\nOutput in {settings.destination.parent.name}/{settings.destination.name}")

    return model

def save_inference(model_name:str, in_channels: int, settings: SettingsTraining):
    # push all datapoints through and save all outputs
    if settings.problem == "2stages":
        model = UNet(in_channels=in_channels).float()

    model.load(model_name, map_location=settings.device)
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

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_raw", type=str, default="dataset_2d_small_1000dp", help="Name of the raw dataset (without inputs)")
    parser.add_argument("--dataset_prep", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--case", type=str, choices=["train", "test", "finetune", "hypertune", "visualize", "iterative", "prep_xhp"], default="train")
    parser.add_argument("--model", type=str, default="default") # required for testing or finetuning
    parser.add_argument("--destination", type=str, default="")
    parser.add_argument("--inputs", type=str, default="gksit") #choices=["gki", "gksi", "pksi", "gks", "gksi100", "ogksi1000", "gksi1000", "pksi100", "pksi1000", "ogksi1000_finetune", "gki100", "t", "gkiab", "gksiab", "gkt"]
    parser.add_argument("--visualize", type=bool, default=False)
    parser.add_argument("--only_prep", type=bool, default=False)
    parser.add_argument("--save_inference", type=bool, default=False)
    parser.add_argument("--problem", type=str, choices=["2stages","parallel","quad"], default="2stages")
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--len_box", type=int, default=256)
    parser.add_argument("--skip_per_dir", type=int, default=256)
    args = parser.parse_args()
    settings = SettingsTraining(**vars(args))

    settings = prepare_data_and_paths(settings)
    if not settings.case == "prep_xhp":
        model = run(settings)
        if args.save_inference:
            save_inference(settings.model, len(args.inputs), settings)