from dataclasses import dataclass, field
import numpy as np
from math import inf
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
import logging
from typing import Dict, Tuple

from data.dataloader import DataLoader
from networks.unet import UNet

# TODO: look at vispy library for plotting 3D data

@dataclass
class DataToVisualize:
    data:np.ndarray
    name:str
    index:int
    imshowargs:Dict = field(default_factory=dict)
    contourargs:Dict = field(default_factory=dict)

    def __post_init__(self):
        self.data = self.data[0,self.index,:,:]
        self.contourargs = {"levels" : np.arange(10.6, 15.6, 0.25), "cmap" : "RdBu_r", "extent" : (0, 1280, 400, 0)} # TODO extent is hardcoded

def plot_sample(model:UNet, dataloader: DataLoader, device:str, amount_plots:int=inf, plot_name:str="default"):

    error = []
    error_mean = []

    if amount_plots > len(dataloader.dataset):
        amount_plots = len(dataloader.dataset)

    for batch_id, (inputs,labels) in enumerate(dataloader):
        len_batch = inputs.shape[0]
        for datapoint_id in range(len_batch):
            # get data
            x = inputs[datapoint_id].to(device)
            x = torch.unsqueeze(x,0)
            y = labels[datapoint_id].to(device)
            y = torch.unsqueeze(y,0)
            model.eval()
            y_out = model(x).to(device)

            # reverse transform for plotting real values
            x = dataloader.dataset.reverse_transform(x)
            y = dataloader.dataset.reverse_transform_temperature(y)
            y_out = dataloader.dataset.reverse_transform_temperature(y_out)

            # calculate error
            error_current = y-y_out
            error.append(abs(error_current.detach()))
            error_mean.append(np.mean(error_current.cpu().numpy()).item())

            # plot temperature true, temperature out, error, physical variables
            temp_max = max(y.max(), y_out.max())
            temp_min = min(y.min(), y_out.min())
            dict_to_plot = {
                "t_true": DataToVisualize(y.detach().cpu(), "Temperature True [°C]", 0, {"vmax":temp_max, "vmin":temp_min}),
                "t_out": DataToVisualize(y_out.detach().cpu(), "Temperature Out [°C]", 0, {"vmax":temp_max, "vmin":temp_min}),
                "error": DataToVisualize(np.abs(error_current.detach().cpu()), "Abs. Error [°C]", 0),
            }
            try:
                physical_vars = dataloader.dataset.datapoints[datapoint_id].inputs.keys()
            except:
                physical_vars = dataloader.dataset[datapoint_id].inputs.keys()
            for index, physical_var in enumerate(physical_vars):
                dict_to_plot[physical_var] = DataToVisualize(x.detach().cpu(), physical_var, index)

            current_id = datapoint_id + batch_id*len_batch
            name_pic = f"runs/{plot_name}_{current_id}"
            _plot_datafields(dict_to_plot, name_pic=name_pic)
            # _plot_isolines(dict_to_plot, name_pic=name_pic)

            logging.info(f"Resulting pictures are at runs/{plot_name}_*")
            if current_id == amount_plots-1:
                break

    max_error = np.max(error[-1].cpu().numpy())
    logging.info("Maximum error: ", max_error)
    return error_mean, max_error

def _plot_datafields(data: Dict[str,DataToVisualize], name_pic:str):
    n_subplots = len(data)
    _, axes = plt.subplots(n_subplots,1,sharex=True,figsize=(20,3*(n_subplots)))

    for index, data_point in enumerate(data.values()):
        plt.sca(axes[index])
        plt.imshow(data_point.data.T, extent=data_point.contourargs["extent"], **data_point.imshowargs)
        plt.gca().invert_yaxis()

        plt.ylabel("x [m]")
        plt.xlabel("y [m]")
        _aligned_colorbar(label=data_point.name)

    plt.suptitle("Datafields: Input, Output, Error")
    plt.savefig(f"{name_pic}.png")
    plt.close()

def _plot_isolines(data: Dict[str,DataToVisualize], name_pic:str):
    # helper function to plot isolines of temperature out
    _, axis = plt.subplots(figsize=(20,5))
    plt.sca(axis)
    data_point = data["t_out"]
    plt.contourf(data_point.data[:,:].T, **data_point.contourargs) 
    plt.gca().invert_yaxis()
    plt.ylabel("x [m]")
    plt.xlabel("y [m]")
    _aligned_colorbar()
    
    plt.suptitle(f"Isolines of Temperature [°C]")
    plt.savefig(f"{name_pic}_isolines.png")
    # plt.savefig(f"{name_pic}.svg")
    plt.close()

## helper functions for plotting
def _aligned_colorbar(*args,**kwargs):
    cax = make_axes_locatable(plt.gca()).append_axes("right",size= 0.3,pad= 0.05)
    plt.colorbar(*args,cax=cax,**kwargs)