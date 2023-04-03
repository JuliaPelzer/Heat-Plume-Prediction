from dataclasses import dataclass, field
import numpy as np
from math import inf
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
import logging
from typing import Dict

from torch.utils.data import DataLoader
from networks.unet import UNet

# TODO: look at vispy library for plotting 3D data


@dataclass
class DataToVisualize:
    data: np.ndarray
    name: str
    imshowargs: Dict = field(default_factory=dict)
    contourargs: Dict = field(default_factory=dict)

    def __post_init__(self):
        # TODO: add reasonable extent, make variable
        extent = (0, 1280, 100, 0)
        self.imshowargs["extent"]  = extent
        self.contourargs["extent"] = extent
        self.contourargs = {"levels": np.arange(
            10.4, 15.6, 0.25), "cmap": "RdBu_r", }


def plot_sample(model: UNet, dataloader: DataLoader, device: str, amount_plots: int = inf, plot_name: str = "default"):

    logging.warning("Plotting...")
    error = []
    error_mean = []

    if amount_plots > len(dataloader.dataset):
        amount_plots = len(dataloader.dataset)

    current_id = 0
    input_norm = dataloader.dataset.dataset.input_norm
    label_norm = dataloader.dataset.dataset.label_norm
    for inputs, labels in dataloader:
        len_batch = inputs.shape[0]
        for datapoint_id in range(len_batch):
            # get data
            x = inputs[datapoint_id].to(device)
            x = torch.unsqueeze(x, 0)
            model.eval()
            y_out = model(x).to(device)

            # reverse transform for plotting real values
            y = labels[datapoint_id]
            x = input_norm.reverse(x.detach().cpu().squeeze())
            y = label_norm.reverse(y.detach().cpu())[0]
            y_out = label_norm.reverse(y_out.detach().cpu()[0])[0]

            # calculate error
            error_current = y-y_out
            error.append(abs(error_current))
            error_mean.append(
                torch.mean(error_current).item())

            # plot temperature true, temperature out, error, physical variables
            temp_max = max(y.max(), y_out.max())
            temp_min = min(y.min(), y_out.min())
            dict_to_plot = {
                "t_true": DataToVisualize(y, "Temperature True [°C]", {"vmax": temp_max, "vmin": temp_min}),
                "t_out": DataToVisualize(y_out, "Temperature Out [°C]", {"vmax": temp_max, "vmin": temp_min}),
                "error": DataToVisualize(torch.abs(error_current), "Abs. Error [°C]"),
            }
            info = dataloader.dataset.dataset.info
            physical_vars = info["Inputs"].keys()
            for physical_var in physical_vars:
                index = info["Inputs"][physical_var]["index"]
                dict_to_plot[physical_var] = DataToVisualize(
                    x[index], physical_var)

            name_pic = f"runs/{plot_name}_{current_id}"
            _plot_datafields(dict_to_plot, name_pic=name_pic)
            _plot_isolines(dict_to_plot, name_pic=name_pic)

            logging.info(f"Resulting pictures are at runs/{plot_name}_*")

            if current_id >= amount_plots-1:
                max_error = np.max(error[-1].cpu().numpy())
                logging.info("Maximum error: ", max_error)
                return error_mean, max_error
            current_id += 1


def _plot_datafields(data: Dict[str, DataToVisualize], name_pic: str):
    n_subplots = len(data)
    _, axes = plt.subplots(n_subplots, 1, sharex=True,
                           figsize=(20, 3*(n_subplots)))

    for index, data_point in enumerate(data.values()):
        plt.sca(axes[index])
        plt.imshow(data_point.data.T, **data_point.imshowargs)
        plt.gca().invert_yaxis()

        plt.ylabel("x [m]")
        plt.xlabel("y [m]")
        _aligned_colorbar(label=data_point.name)

    plt.suptitle("Datafields: Input, Output, Error")
    plt.savefig(f"{name_pic}.png")
    plt.close()

def _plot_isolines(data: Dict[str, DataToVisualize], name_pic: str):
    # helper function to plot isolines of temperature out
    _, axis = plt.subplots(figsize=(20, 3)) #TODO (20,5)
    plt.sca(axis)
    data_point = data["t_out"]
    plt.contourf(data_point.data.T, **data_point.contourargs)
    plt.gca().invert_yaxis()
    plt.ylabel("x [m]")
    plt.xlabel("y [m]")
    _aligned_colorbar()

    plt.suptitle(f"Isolines of Temperature [°C]")
    plt.savefig(f"{name_pic}_isolines.png")
    # plt.savefig(f"{name_pic}.svg")
    plt.close()


def _aligned_colorbar(*args, **kwargs):
    cax = make_axes_locatable(plt.gca()).append_axes(
        "right", size=0.3, pad=0.05)
    plt.colorbar(*args, cax=cax, **kwargs)
