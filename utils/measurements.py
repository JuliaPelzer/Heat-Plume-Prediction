import torch
from torch.nn import MSELoss, modules
from torch.utils.data import DataLoader
import os
import time
import yaml
from typing import Dict
import matplotlib.pyplot as plt

from networks.unet import UNet
from solver import Solver
from data_stuff.utils import SettingsTraining

def measure_len_width_1K_isoline(data: Dict[str, "DataToVisualize"]):
    ''' 
    function (for paper23) to measure the length and width of the 1K-isoline;
    prints the values for usage in ipynb
    '''
    
    lengths = {}
    widths = {}
    T_gwf = 10.6

    _, axes = plt.subplots(4, 1, sharex=True)
    for index, key in enumerate(["t_true", "t_out"]):
        plt.sca(axes[index])
        datapoint = data[key]
        datapoint.data = torch.flip(datapoint.data, dims=[1])
        left_bound, right_bound = 1280, 0
        upper_bound, lower_bound = 0, 80
        if datapoint.data.max() > T_gwf + 1:
            levels = [T_gwf + 1] 
            CS = plt.contour(datapoint.data.T, levels=levels, cmap='Pastel1', extent=(0,1280,80,0))

            # calc maximum width and length of 1K-isoline
            for level in CS.allsegs:
                for seg in level:
                    right_bound = max(right_bound, seg[:,0].max())
                    left_bound = min(left_bound, seg[:,0].min())
                    upper_bound = max(upper_bound, seg[:,1].max())
                    lower_bound = min(lower_bound, seg[:,1].min())
        lengths[key] = max(right_bound - left_bound, 0)
        widths[key] = max(upper_bound - lower_bound, 0)
        print(f"lengths_{key[2:]}.append({lengths[key]})")
        print(f"widths_{key[2:]}.append({widths[key]})")
        print(f"max_temps_{key[2:]}.append({datapoint.data.max()})")
        plt.sca(axes[index+2])
    plt.close("all")
    return lengths, widths

def measure_loss(model: UNet, dataloader: DataLoader, device: str, loss_func: modules.loss._Loss = MSELoss()):

    norm = dataloader.dataset.dataset.norm
    model.eval()
    mse_loss = 0.0
    mse_closs = 0.0
    mae_loss = 0.0
    mae_closs = 0.0

    for x, y in dataloader: # batchwise
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x).to(device)
        mse_loss += loss_func(y_pred, y).detach().item()
        mae_loss = torch.mean(torch.abs(y_pred - y)).detach().item()

        y = torch.swapaxes(y, 0, 1)
        y_pred = torch.swapaxes(y_pred, 0, 1)
        y = norm.reverse(y.detach().cpu(),"Labels")
        y_pred = norm.reverse(y_pred.detach().cpu(),"Labels")
        mse_closs += loss_func(y_pred, y).detach().item()
        mae_closs = torch.mean(torch.abs(y_pred - y)).detach().item()
        
    mse_loss /= len(dataloader)
    mse_closs /= len(dataloader)
    mae_loss /= len(dataloader)
    mae_closs /= len(dataloader)

    return {"mean squared error": mse_loss, "mean squared error in [°C^2]": mse_closs, 
            "mean absolute error": mae_loss, "mean absolute error in [°C]": mae_closs}

def save_all_measurements(settings:SettingsTraining, len_dataset, times, solver:Solver=None, errors:Dict={}):
    with open(os.path.join(os.getcwd(), "runs", settings.destination, f"measurements_{settings.case}.yaml"), "w") as f:
        for key, value in times.items():
            f.write(f"{key}: {value}\n")
        f.write(f"timestamp of end: {time.ctime()}\n")
        f.write(f"duration of whole process including visualisation in seconds: {(times['time_end']-times['time_begin'])}\n")
        if settings.case in ["train", "finetune"]: 
            f.write(f"duration of initializations in seconds: {times['time_initializations']-times['time_begin']}\n")
            f.write(f"duration of training in seconds: {times['time_training']-times['time_initializations']}\n")

        f.write(f"input params: {settings.inputs}\n")
        f.write(f"dataset location: {settings.datasets_dir}\n")
        f.write(f"dataset name: {settings.dataset_raw}\n")
        f.write(f"case: {settings.case}\n")
        f.write(f"number of datapoints: {len_dataset}\n")
        f.write(f"name_destination_folder: {settings.destination}\n")
        f.write(f"number epochs: {settings.epochs}\n")

        for key, value in errors.items():
            f.write(f"{key}: {value}\n")
        if settings.case in ["test", "finetune"]:
            f.write(f"path to pretrained model: {settings.model}\n")
        if settings.case in ["train", "finetune"]: 
            f.write(f"best model found after epoch: {solver.best_model_params['epoch']}\n")
            f.write(f"best model found with val loss: {solver.best_model_params['loss']}\n")
            f.write(f"best model found with train loss: {solver.best_model_params['train loss']}\n")
            f.write(f"best model found with val RMSE: {solver.best_model_params['val RMSE']}\n")
            f.write(f"best model found with train RMSE: {solver.best_model_params['train RMSE']}\n")
            f.write(f"best model found after training time in seconds: {solver.best_model_params['training time in sec']}\n")

    print(f"Measurements saved")