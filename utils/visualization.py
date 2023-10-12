import logging
import time
from dataclasses import dataclass, field
from math import inf
from typing import Dict

import matplotlib as mpl
mpl.use('pgf')
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.utils.data import DataLoader

from networks.unet import UNet

mpl.rcParams.update({'figure.max_open_warning': 0})
plt.rcParams['figure.figsize'] = [8, 2.5]

# TODO: look at vispy library for plotting 3D data

@dataclass
class DataToVisualize:
    data: np.ndarray
    name: str
    extent_highs :tuple = (1280,100) # x,y in meters
    imshowargs: Dict = field(default_factory=dict)
    contourfargs: Dict = field(default_factory=dict)
    contourargs: Dict = field(default_factory=dict)

    def __post_init__(self):
        extent = (0,int(self.extent_highs[0]),int(self.extent_highs[1]),0)

        self.imshowargs = {"cmap": "RdBu_r", 
                           "extent": extent}

        self.contourfargs = {"levels": np.arange(10.4, 16, 0.25), 
                             "cmap": "RdBu_r", 
                             "extent": extent}
        
        T_gwf = 10.6
        T_inj_diff = 5.0
        self.contourargs = {"levels" : [np.round(T_gwf + 1, 1)],
                            "cmap" : "Pastel1", 
                            "extent": extent}

        if self.name == "Liquid Pressure [Pa]":
            self.name = "Pressure in [Pa]"
        elif self.name == "Material ID":
            self.name = "Position of the heatpump in [-]"
        elif self.name == "Permeability X [m^2]":
            self.name = "Permeability in [m$^2$]"
        elif self.name == "SDF":
            self.name = "SDF-transformed position in [-]"
    
def plot_sample(model: UNet, dataloader: DataLoader, device: str, amount_plots: int = inf, plot_name: str = "default", pic_format: str = "png"):
    logging.warning("Plotting...")

    if amount_plots > len(dataloader.dataset):
        amount_plots = len(dataloader.dataset)

    norm = dataloader.dataset.dataset.norm
    info = dataloader.dataset.dataset.info
    model.eval()
    current_id = 0

    for inputs, labels in dataloader:
        len_batch = inputs.shape[0]
        for datapoint_id in range(len_batch):
            # get data
            x = inputs[datapoint_id].to(device)
            x = torch.unsqueeze(x, 0)
            y = labels[datapoint_id]

            # inference
            y_out = model(x).to(device)

            # reverse transform for plotting real values
            x = norm.reverse(x.detach().cpu().squeeze(), "Inputs")
            y = norm.reverse(y.detach().cpu(),"Labels")[0]
            y_out = norm.reverse(y_out.detach().cpu()[0],"Labels")[0]

            # plot temperature true, temperature out, error, physical variables
            temp_max = max(y.max(), y_out.max())
            temp_min = min(y.min(), y_out.min())
            extent_highs = (np.array(info["CellsSize"][:2]) * y.shape)
            dict_to_plot = {
                "t_true": DataToVisualize(y, "Label: Temperature in [°C]",extent_highs, {"vmax": temp_max, "vmin": temp_min}),
                "t_out": DataToVisualize(y_out, "Prediction: Temperature in [°C]",extent_highs, {"vmax": temp_max, "vmin": temp_min}),
                "error": DataToVisualize(torch.abs(y-y_out), "Absolute error in [°C]",extent_highs),
            }
            physical_vars = info["Inputs"].keys()
            for physical_var in physical_vars:
                index = info["Inputs"][physical_var]["index"]
                dict_to_plot[physical_var] = DataToVisualize(
                    x[index], physical_var,extent_highs)

            name_pic = f"runs/{plot_name}_{current_id}"
            figsize_x = extent_highs[0]/extent_highs[1]*3
            _plot_datafields(dict_to_plot, name_pic=name_pic, figsize_x=figsize_x, pic_format=pic_format)
            # _plot_isolines(dict_to_plot, name_pic=name_pic, figsize_x=figsize_x, pic_format=pic_format)
            # _isolines_measurements(dict_to_plot, name_pic=name_pic, figsize_x=figsize_x)
            # _plot_temperature_field(dict_to_plot, name_pic=name_pic, figsize_x=figsize_x, pic_format=pic_format)

            if current_id >= amount_plots-1:
                return None
            current_id += 1

def infer_all_and_summed_pic(model: UNet, dataloader: DataLoader, device: str):
    # sum inference time and error over all datapoints
    
    norm = dataloader.dataset.dataset.norm
    model.eval()

    current_id = 0
    avg_inference_time = 0
    summed_error_pic = torch.zeros_like(torch.Tensor(dataloader.dataset[0][0][0]))

    for inputs, labels in dataloader:
        len_batch = inputs.shape[0]
        for datapoint_id in range(len_batch):
            # get data
            start_time = time.perf_counter()
            x = inputs[datapoint_id].to(device)
            x = torch.unsqueeze(x, 0)
            y_out = model(x).to(device)
            y = labels[datapoint_id]

            # reverse transform for plotting real values
            x = norm.reverse(x.detach().cpu().squeeze(), "Inputs")
            y = norm.reverse(y.detach().cpu(),"Labels")[0]
            y_out = norm.reverse(y_out.detach().cpu()[0],"Labels")[0]
            avg_inference_time += (time.perf_counter() - start_time)
            summed_error_pic += abs(y-y_out)

            current_id += 1

    avg_inference_time = avg_inference_time / current_id
    summed_error_pic = summed_error_pic / current_id
    return avg_inference_time, summed_error_pic

def plt_avg_error_cellwise(model: UNet, dataloader: DataLoader, device: str, plot_name: str = "default"):
    # plot avg error cellwise AND return time measurements for inference

    avg_inference_time, summed_error_pic = infer_all_and_summed_pic(model, dataloader, device)

    info = dataloader.dataset.dataset.info
    extent_highs = (np.array(info["CellsSize"][:2]) * dataloader.dataset[0][0][0].shape)
    _plot_avg_error(summed_error_pic, plot_name, extent_highs)

    return avg_inference_time

def _plot_avg_error(data, plot_name:str, extent_highs:tuple):
    extent = (0,int(extent_highs[0]),int(extent_highs[1]),0)
    plt.figure()
    plt.imshow(data.T, cmap="RdBu_r", extent=extent)
    plt.gca().invert_yaxis()
    plt.ylabel("x [m]")
    plt.xlabel("y [m]")
    plt.title("Pixelwise averaged error [°C]")
    _aligned_colorbar()
    plt.savefig(f"runs/{plot_name}_pixelwise_avg_error.pgf", format="pgf")
    plt.savefig(f"runs/{plot_name}_pixelwise_avg_error.png")

def _plot_datafields(data: Dict[str, DataToVisualize], name_pic: str, figsize_x: float = 38.4, pic_format: str = "png"):
    n_subplots = len(data)
    fig, axes = plt.subplots(n_subplots, 1, sharex=True) #, figsize=(figsize_x, 3*(n_subplots)))
    fig.set_figheight(n_subplots)
    for index, (name, datapoint) in enumerate(data.items()):
        plt.sca(axes[index])
        plt.title(datapoint.name)
        if name in ["t_true", "t_out"]:  
            CS = plt.contour(torch.flip(datapoint.data, dims=[1]).T, **datapoint.contourargs)
            plt.clabel(CS, inline=1, fontsize=10)

        plt.imshow(datapoint.data.T, **datapoint.imshowargs)
        plt.gca().invert_yaxis()

        plt.ylabel("x [m]")
        _aligned_colorbar()

    plt.sca(axes[-1])
    plt.xlabel("y [m]")
    plt.tight_layout()
    # plt.suptitle("Datafields: Inputs, Output, Error")
    plt.savefig(f"{name_pic}.{pic_format}", format=pic_format)

def _plot_isolines(data: Dict[str, DataToVisualize], name_pic: str, figsize_x: float = 38.4, pic_format: str = "png"):
    # helper function to plot isolines of temperature out
    
    if "Original Temperature [C]" in data.keys():
        num_subplots = 3
    else:
        num_subplots = 2
    fig, axes = plt.subplots(num_subplots, 1, sharex=True) #, figsize=(figsize_x, 3*2))
    fig.set_figheight(num_subplots)

    for index, name in enumerate(["t_true", "t_out", "Original Temperature [C]"]):
        try:
            plt.sca(axes[index])
            datapoint = data[name]
            datapoint.data = torch.flip(datapoint.data, dims=[1])
            plt.title("Isolines of "+datapoint.name)
            plt.contourf(datapoint.data.T, **datapoint.contourfargs)
            plt.ylabel("x [m]")
            _aligned_colorbar(ticks=[11.6, 15.6])
        except:
            pass

    plt.sca(axes[-1])
    plt.xlabel("y [m]")
    plt.tight_layout()

    plt.savefig(f"{name_pic}_isolines.{pic_format}", format=pic_format)

def _isolines_measurements(data: Dict[str, DataToVisualize]):
    # helper function to plot isolines of temperature out
    
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
        plt.imshow(datapoint.data.T, extent=(0,1280,80,0))
    plt.show()
    plt.close("all")
    return lengths, widths

def _plot_temperature_field(data: Dict[str, DataToVisualize], name_pic:str, figsize_x: float = 38.4, pic_format: str = "png"):
    """
    Plot the temperature field. 
    almost-copy of other_models/analytical_models/utils_and_visu
    """
    _, axes = plt.subplots(3,1,sharex=True,figsize=(figsize_x, 3*3))
    
    for index, name in enumerate(["t_true", "t_out", "error"]):
            plt.sca(axes[index])
            datapoint = data[name]
            if name=="error": datapoint.contourargs["levels"] = [level - 10.6 for level in datapoint.contourargs["levels"]]
            CS = plt.contour(torch.flip(datapoint.data, dims=[1]).T, **datapoint.contourargs)
            plt.clabel(CS, inline=1, fontsize=10)
            plt.imshow(datapoint.data.T, **datapoint.imshowargs)
            plt.gca().invert_yaxis()
            plt.xlabel("y [m]")
            plt.ylabel("x [m]")
            _aligned_colorbar(label=datapoint.name)

    T_gwf_plus1, T_gwf_plusdiff = datapoint.contourargs["levels"]
    plt.suptitle(f"Temperature field and isolines of {T_gwf_plus1} and {T_gwf_plusdiff} °C")
    plt.savefig(f"{name_pic}_combined.{pic_format}", format=pic_format)

def _aligned_colorbar(*args, **kwargs):
    cax = make_axes_locatable(plt.gca()).append_axes(
        "right", size=0.3, pad=0.05)
    plt.colorbar(*args, cax=cax, **kwargs)
