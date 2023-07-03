import logging
from datetime import datetime
from typing import Dict
from math import inf
import numpy as np
import matplotlib.pyplot as plt
import torch
from dataclasses import dataclass, field
from mpl_toolkits.axes_grid1 import make_axes_locatable
from line_profiler_decorator import profiler

from torch.utils.data import DataLoader
from networks.unet import UNet

import matplotlib as mpl
mpl.rcParams.update({'figure.max_open_warning': 0})

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

        self.contourfargs = {"levels": np.arange(10.4, 15.6, 0.25), 
                             "cmap": "RdBu_r", 
                             "extent": extent}
        
        T_gwf = 10.6
        T_inj_diff = 5.0
        self.contourargs = {"levels" : [np.round(T_gwf + 1, 1), np.round(T_gwf + T_inj_diff, 1)],
                            "cmap" : "Pastel1", 
                            "extent": extent}

def plot_sample(model: UNet, dataloader: DataLoader, device: str, amount_plots: int = inf, plot_name: str = "default"):

    logging.warning("Plotting...")
    error = []
    error_mean = []

    if amount_plots > len(dataloader.dataset):
        amount_plots = len(dataloader.dataset)

    current_id = 0
    norm = dataloader.dataset.dataset.norm
    for inputs, labels in dataloader:
        len_batch = inputs.shape[0]
        for datapoint_id in range(len_batch):
            # get data
            start_time = datetime.now()
            x = inputs[datapoint_id].to(device)
            x = torch.unsqueeze(x, 0)
            model.eval()
            y_out = model(x).to(device)

            # reverse transform for plotting real values
            y = labels[datapoint_id]
            x = norm.reverse(x.detach().cpu().squeeze(), "Inputs")
            y = norm.reverse(y.detach().cpu(),"Labels")[0]
            y_out = norm.reverse(y_out.detach().cpu()[0],"Labels")[0]
            logging.info(datapoint_id)
            logging.info(f"Time of inference: {(datetime.now() - start_time).total_seconds()}s")
            loc_max = y_out.argmax()
            logging.info(f"Max temp: {y_out.max()} at {(loc_max%100, loc_max//100%1280, loc_max//100//1280%5)}")

            # calculate error
            error_current = y-y_out
            error.append(abs(error_current))
            error_mean.append(
                torch.mean(error_current).item())

            # plot temperature true, temperature out, error, physical variables
            temp_max = max(y.max(), y_out.max())
            temp_min = min(y.min(), y_out.min())
            info = dataloader.dataset.dataset.info
            extent_highs = (np.array(info["CellsSize"][:2]) * y.shape)
            dict_to_plot = {
                "t_true": DataToVisualize(y, "Temperature True [°C]",extent_highs, {"vmax": temp_max, "vmin": temp_min}),
                "t_out": DataToVisualize(y_out, "Temperature Out [°C]",extent_highs, {"vmax": temp_max, "vmin": temp_min}),
                "error": DataToVisualize(torch.abs(error_current), "Abs. Error [°C]",extent_highs),
            }
            physical_vars = info["Inputs"].keys()
            for physical_var in physical_vars:
                index = info["Inputs"][physical_var]["index"]
                dict_to_plot[physical_var] = DataToVisualize(
                    x[index], physical_var,extent_highs)

            name_pic = f"runs/{plot_name}_{current_id}"
            figsize_x = extent_highs[0]/extent_highs[1]*3
            _plot_datafields(dict_to_plot, name_pic=name_pic, figsize_x=figsize_x)
            _plot_isolines(dict_to_plot, name_pic=name_pic, figsize_x=figsize_x)
            # _plot_temperature_field(dict_to_plot, name_pic=name_pic, figsize_x=figsize_x)

            logging.info(f"Resulting pictures are at runs/{plot_name}_*")

            # if (current_id > 0 and current_id % 6 == 0) or current_id >= amount_plots-1:
            #     plt.close("all")

            if current_id >= amount_plots-1:
                max_error = np.max(error[-1].cpu().numpy())
                logging.info("Maximum error: ", max_error)
                return error_mean, max_error
            current_id += 1

# @profiler        
def _plot_datafields(data: Dict[str, DataToVisualize], name_pic: str, figsize_x: float = 38.4):
    n_subplots = len(data)
    _, axes = plt.subplots(n_subplots, 1, sharex=True,
                           figsize=(figsize_x, 3*(n_subplots)))

    for index, (name, datapoint) in enumerate(data.items()):
        plt.sca(axes[index])
        if name in ["t_true", "t_out", "error"]:  
            if name=="error": datapoint.contourargs["levels"] = [level - 10.6 for level in datapoint.contourargs["levels"]] 
            CS = plt.contour(torch.flip(datapoint.data, dims=[1]).T, **datapoint.contourargs)
            plt.clabel(CS, inline=1, fontsize=10)

        plt.imshow(datapoint.data.T, **datapoint.imshowargs)
        plt.gca().invert_yaxis()

        plt.ylabel("x [m]")
        plt.xlabel("y [m]")
        _aligned_colorbar(label=datapoint.name)

    plt.suptitle("Datafields: Inputs, Output, Error")
    plt.savefig(f"{name_pic}.png")
    # plt.savefig(f"{name_pic}.svg")

def _plot_isolines(data: Dict[str, DataToVisualize], name_pic: str, figsize_x: float = 38.4):
    # helper function to plot isolines of temperature out
    
    if "Original Temperature [C]" in data.keys():
        num_subplots = 3
    else:
        num_subplots = 2
    _, axes = plt.subplots(num_subplots, 1, sharex=True, figsize=(figsize_x, 3*2))

    for index, name in enumerate(["t_true", "t_out", "Original Temperature [C]"]):
        try:
            plt.sca(axes[index])
            datapoint = data[name]
            datapoint.data = torch.flip(datapoint.data, dims=[1])
            plt.contourf(datapoint.data.T, **datapoint.contourfargs)
            plt.ylabel("x [m]")
            plt.xlabel("y [m]")
            _aligned_colorbar(label=datapoint.name)
        except:
            pass

    plt.suptitle(f"Isolines of Temperature [°C]")
    plt.savefig(f"{name_pic}_isolines.png")
    # plt.savefig(f"{name_pic}.svg")

def _plot_temperature_field(data: Dict[str, DataToVisualize], name_pic:str, figsize_x: float = 38.4):
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
    plt.savefig(f"{name_pic}_combined.png")
    plt.savefig(f"{name_pic}_combined.svg")

def _aligned_colorbar(*args, **kwargs):
    cax = make_axes_locatable(plt.gca()).append_axes(
        "right", size=0.3, pad=0.05)
    plt.colorbar(*args, cax=cax, **kwargs)
