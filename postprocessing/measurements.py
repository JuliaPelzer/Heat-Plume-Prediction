import argparse
import time
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import torch
from torch.nn import MSELoss, L1Loss, modules
# from torcheval.metrics import MeanSquaredError # TODO for all, R2Score
from torch.utils.data import DataLoader

from processing.networks.unet import UNet
from processing.solver import Solver


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

    try:
        norm = dataloader.dataset.norm
    except AttributeError:
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
        mae_loss += torch.mean(torch.abs(y_pred - y)).detach().item()

        y = torch.swapaxes(y, 0, 1)
        y_pred = torch.swapaxes(y_pred, 0, 1)
        y = norm.reverse(y.detach().cpu(),"Labels")
        y_pred = norm.reverse(y_pred.detach().cpu(),"Labels")
        mse_closs += loss_func(y_pred, y).detach().item()
        mae_closs += torch.mean(torch.abs(y_pred - y)).detach().item()
        
    mse_loss /= len(dataloader)
    mse_closs /= len(dataloader)
    mae_loss /= len(dataloader)
    mae_closs /= len(dataloader)

    return {"mean squared error": mse_loss, "mean squared error in [°C^2]": mse_closs, 
            "mean absolute error": mae_loss, "mean absolute error in [°C]": mae_closs,
            }

def measure_losses_paper24(model: UNet, dataloaders: Dict[str, DataLoader], args: dict, vT_case: str = "temperature"):
    '''
    function to measure the losses for the paper24
    ATTENTION! not robust, expects vT-case to be "temperature" or "velocities" and
    sets the number of outputs accordingly (1 or 2)
    also: calculates the pbt_threshold only for temperature
    '''
    if vT_case == "temperature":
        pbt_thresholds = [0.1, 1] # [°C] # only relevant for temperature
    
    device = args["device"]
    if args["problem"] == "allin1":
        norm = dataloaders["train"].dataset.norm
        output_channels = dataloaders["train"].dataset.output_channels
    elif args["problem"] in ["1hp", "2stages"]:
        norm = dataloaders["train"].dataset.dataset.norm
        output_channels = dataloaders["train"].dataset.dataset.output_channels
    model.eval()
    results = {}

    for case, dataloader in dataloaders.items():
        mse_loss = torch.Tensor([0.0,] * output_channels)
        mae_closs = torch.Tensor([0.0,] * output_channels)
        rmse_closs = torch.Tensor([0.0,] * output_channels)
        if vT_case == "temperature":
            pbt_closs = torch.Tensor([0.0,] * len(pbt_thresholds))

        for x, y in dataloader:
            start_inference = time.perf_counter()
            x = x.to(device).detach() # B,C,H,W
            y = y.to(device).detach()
            y_pred = model(x).to(device).detach()
            time_inference = time.perf_counter() - start_inference
            required_size = y_pred.shape[2:]
            start_pos = ((y.shape[2] - required_size[0])//2, (y.shape[3] - required_size[1])//2)
            y = y[:, :, start_pos[0]:start_pos[0]+required_size[0], start_pos[1]:start_pos[1]+required_size[1]]

            # normed losses
            for channel in range(y_pred.shape[1]):
                mse_loss[channel] += MSELoss(reduction="sum")(y_pred[:,channel], y[:,channel]).item()

            # reverse norm -> values in original units and scales
            y = torch.swapaxes(y, 0, 1) # for norm, C needs to be first
            y_pred = torch.swapaxes(y_pred, 0, 1)
            y = norm.reverse(y.cpu(),"Labels")
            y_pred = norm.reverse(y_pred.cpu(),"Labels")
            y = torch.swapaxes(y, 0, 1)
            y_pred = torch.swapaxes(y_pred, 0, 1)

            # losses in Celsius
            for channel in range(y_pred.shape[1]):
                mae_closs[channel] += L1Loss(reduction="sum")(y_pred[:,channel], y[:,channel]).item()
                rmse_closs[channel] += MSELoss(reduction="sum")(y_pred[:,channel], y[:,channel]).item()

            if vT_case == "temperature":
                # count all pixels where the difference is larger than threshold, then average over this batch + domain
                for idx in range(len(pbt_thresholds)):
                    pbt_closs[idx] += (torch.sum(torch.abs(y_pred[:,0] - y[:,0]) > pbt_thresholds[idx])).item()

        # average over all batches
        no_datapoints = len(dataloaders[case].dataset)
        domain_size = y_pred.shape[2] * y_pred.shape[3]
        mse_loss /= (no_datapoints * domain_size)
        mae_closs /= (no_datapoints * domain_size)
        rmse_closs /=  (no_datapoints * domain_size)
        rmse_closs = torch.sqrt(rmse_closs)
        if vT_case == "temperature":
            pbt_closs /= (no_datapoints * domain_size) 
            pbt_closs *= 100 # value close to 0% is good, close to 100% is bad

        output_mse = ["{:.2e}".format(mse) for mse in mse_loss]
        output_mae = ["{:.2e}".format(mae) for mae in mae_closs]
        output_rmse = ["{:.2e}".format(rmse) for rmse in rmse_closs]
        if vT_case == "temperature":
            output_pbt = ["{:.2f} above {:.1f}C".format(pbt, threshold) for threshold, pbt in zip(pbt_thresholds, pbt_closs)]

        if vT_case == "temperature":
            unit = "C"
        else:
            unit = "m/year"
        results[case] = {"MSE [-] per channel": output_mse, f"MAE [{unit}] per channel": output_mae, f"RMSE [{unit}] per channel": output_rmse,
                         "time inference [s]": time_inference} 
        
        if vT_case == "temperature":
            results[case]["PAT (percentage above threshold in [%]"] = output_pbt
    return results

# collect metric of test data for all models
def collect_metric(metrics, dataset, metric_name, idx=0):
    metric_values = {}
    for model_name, metric in metrics.items():
        metric_values[model_name] = float(metric[dataset][metric_name][idx])
    return metric_values

def plot_metrics_velocities(metrics, dataset, metric_name, save_bool=False, destination:Path=None):
    metric_values_vx = collect_metric(metrics, dataset, metric_name, 0)
    metric_values_vy = collect_metric(metrics, dataset, metric_name, 1)

    model_names =       list(metric_values_vx.keys())
    metric_values_vx =  list(metric_values_vx.values())
    metric_values_vy =  list(metric_values_vy.values())
    metric_min_vx = min(metric_values_vx)
    metric_min_vy = min(metric_values_vy)
    model_min_vx = model_names[metric_values_vx.index(metric_min_vx)]
    model_min_vy = model_names[metric_values_vy.index(metric_min_vy)]


    plt.figure()
    # rm "per channel" from metric_name
    metric_name = metric_name[:-12]
    plt.plot(metric_values_vx,  "bx", label=f"{metric_name} for vx")
    plt.plot(metric_values_vy,  "rx", label=f"{metric_name} for vy")
    plt.scatter(model_names.index(model_min_vx), metric_min_vx, 100, c="b", label=f"optimal {metric_name} for vx")
    plt.scatter(model_names.index(model_min_vy), metric_min_vy, 100, c="r", label=f"optimal {metric_name} for vy")
    plt.legend()
    plt.xticks(range(len(model_names)), model_names, rotation=90)
    plt.grid()
    plt.xlabel("Model")
    plt.ylabel(metric_name)
    plt.title(f"{dataset} {metric_name} of vx and vy for all models")
    plt.tight_layout()
    if save_bool:
        plt.savefig(destination/f"{dataset}_{metric_name[:-6]}_all_models.png")
    plt.show()


def plot_metrics_temperature(metrics, dataset, metric_name, save_bool=False, destination:Path=None):
    metric_values_T = collect_metric(metrics, dataset, metric_name, 0)

    model_names = list(metric_values_T.keys())
    metric_values_T = list(metric_values_T.values())
    metric_min_T = min(metric_values_T)
    model_min_T = model_names[metric_values_T.index(metric_min_T)]


    plt.figure()
    # rm "per channel" from metric_name
    metric_name = metric_name[:-12]
    plt.plot(metric_values_T,  "bx", label=f"{metric_name} for T")
    plt.scatter(model_names.index(model_min_T), metric_min_T, 100, c="b", label=f"optimal {metric_name} for T")
    plt.legend()
    plt.xticks(range(len(model_names)), model_names, rotation=90)
    plt.grid()
    plt.xlabel("Model")
    plt.ylabel(metric_name)
    plt.title(f"{dataset} {metric_name} of T for all models")
    plt.tight_layout()
    if save_bool:
        plt.savefig(destination/f"{dataset}_{metric_name[:-4]}_all_models.png")
    plt.show()

def measure_additional_losses(model: UNet, dataloaders: Dict[str, DataLoader], device: str, summed_error_pic: torch.Tensor = None, settings: argparse.Namespace = None):
    
    norm = dataloaders["train"].dataset.dataset.norm
    model.eval()
    results = {}
    for case, dataloader in dataloaders.items():
        vars = 0.0

        for x, y in dataloader: # batchwise
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x).to(device)
            y = torch.swapaxes(y, 0, 1)
            y_pred = torch.swapaxes(y_pred, 0, 1)
            y = norm.reverse(y.detach().cpu(),"Labels")
            y_pred = norm.reverse(y_pred.detach().cpu(),"Labels")
            vars += torch.mean(torch.pow(abs((y_pred - y) - summed_error_pic),2)).detach()
            
        vars /= len(dataloader)
        stds = torch.sqrt(vars)
        print("stds", stds, "vars", vars)

        results[case] = {"variance in [°C]": vars, "standard deviation in [°C]": stds,}

    if settings is not None:
        with open(settings.destination/"2ndreview_measurements_additional.yaml", "w") as f:
            for key, value in results.items():
                f.write(f"{key}: {value}\n")

def save_all_measurements(settings:argparse.Namespace, len_dataset, times, solver:Solver=None, errors:Dict={}):
    with open(Path.cwd() / "runs" / settings.destination / f"measurements_{settings.case}.yaml", "w") as f:
        for key, value in times.items():
            f.write(f"{key}: {value}\n")
        f.write(f"timestamp of end: {time.ctime()}\n")
        f.write(f"duration of whole process including visualisation in seconds: {(times['time_end']-times['time_begin'])}\n")
        if settings.case in ["train", "finetune"]: 
            f.write(f"duration of initializations in seconds: {times['time_initializations']-times['time_begin']}\n")
            f.write(f"duration of training in seconds: {times['time_training']-times['time_initializations']}\n")

        f.write(f"input params: {settings.inputs}\n")
        f.write(f"dataset location: {settings.dataset_prep.parent}\n")
        f.write(f"dataset name: {settings.dataset_train}\n")
        f.write(f"dataset val: {settings.dataset_val}\n")
        f.write(f"dataset test: {settings.dataset_test}\n")
        f.write(f"case: {settings.case}\n")
        f.write(f"number of test datapoints: {len_dataset}\n")
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

    if settings.case in ["train", "finetune"]:  
        with open(Path.cwd() / "runs" / settings.destination.parent / f"measurements_all_runs.csv", "a") as f:
            #name, settings.epochs, epoch, val loss, train loss, val rmse, train rmse, train time
            f.write(f"{settings.destination.name},{settings.epochs},{solver.best_model_params['epoch']}, {round(solver.best_model_params['loss'],6)}, {round(solver.best_model_params['train loss'],6)}, {round(solver.best_model_params['val RMSE'],6)}, {round(solver.best_model_params['train RMSE'],6)}, {round(solver.best_model_params['training time in sec'],6)}\n")

    print(f"Measurements saved")
