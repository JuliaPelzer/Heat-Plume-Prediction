import time
from dataclasses import dataclass, field
from math import inf
from typing import Dict

# import matplotlib as mpl
# mpl.use('pgf')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import Module, MSELoss, modules
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.utils.data import DataLoader

from preprocessing.data_stuff.transforms import NormalizeTransform
from processing.networks.unet import UNet
from utils.utils_data import SettingsTraining

# plt.rcParams['figure.figsize'] = [16, 5]

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
    
def visualizations(model: UNet, dataloader: DataLoader, settings: SettingsTraining, amount_datapoints_to_visu: int = inf, plot_path: str = "default", pic_format: str = "png", different_datasets: bool = False):
    print("Visualizing...", end="\r")

    if amount_datapoints_to_visu > len(dataloader.dataset):
        amount_datapoints_to_visu = len(dataloader.dataset)

    try:
        norm = dataloader.dataset.norm
        info = dataloader.dataset.info
    except AttributeError:
        norm = dataloader.dataset.dataset.norm
        info = dataloader.dataset.dataset.info
    model.eval()
    settings_pic = {"format": pic_format,
                    "dpi": 600,}

    current_id = 0
    for inputs, labels in dataloader:
        len_batch = inputs.shape[0]
        for datapoint_id in range(len_batch):
            name_pic = f"{plot_path}_{current_id}"

            x = torch.unsqueeze(inputs[datapoint_id].to(settings.device), 0)
            y = labels[datapoint_id]
            y_out = model(x).to(settings.device)

            x, y, y_out = reverse_norm_one_dp(x, y, y_out, norm)
            dict_to_plot = prepare_data_to_plot(x, y, y_out, info)

            if different_datasets:
                if settings.case=="test":
                    plot_datafields(dict_to_plot, name_pic, settings_pic, only_inner=True)
                plot_datafields(dict_to_plot, name_pic, settings_pic, only_inner=False, plot_all_in_1_pic=False)
            else:
                plot_datafields(dict_to_plot, name_pic, settings_pic)

            # plot_isolines(dict_to_plot, name_pic, settings_pic)
            # measure_len_width_1K_isoline(dict_to_plot)

            if current_id >= amount_datapoints_to_visu-1:
                return None
            current_id += 1

def reverse_norm_one_dp(x: torch.Tensor, y: torch.Tensor, y_out:torch.Tensor, norm: NormalizeTransform):
    # reverse transform for plotting real values
    x = norm.reverse(x.detach().cpu().squeeze(0), "Inputs")
    y = norm.reverse(y.detach().cpu(),"Labels")[0]
    y_out = norm.reverse(y_out.detach().cpu()[0],"Labels")[0]
    return x, y, y_out

def prepare_data_to_plot(x: torch.Tensor, y: torch.Tensor, y_out:torch.Tensor, info: dict):
    # prepare data of temperature true, temperature out, error, physical variables (inputs)
    temp_max = max(y.max(), y_out.max())
    temp_min = min(y.min(), y_out.min())
    extent_highs = (np.array(info["CellsSize"][:2]) * x.shape[-2:])

    required_size = y_out.shape[0]
    start_pos = (y.shape[0] - required_size)//2
    y_red = y[start_pos:start_pos+required_size, :]

    dict_to_plot = {
        "t_true": DataToVisualize(y_red, "Label: Temperature in [°C]", extent_highs, {"vmax": temp_max, "vmin": temp_min}),
        "t_out": DataToVisualize(y_out, "Prediction: Temperature in [°C]", extent_highs, {"vmax": temp_max, "vmin": temp_min}),
        "error": DataToVisualize(torch.abs(y_red-y_out), "Absolute error in [°C]", extent_highs),
    }
    inputs = info["Inputs"].keys()
    for input in inputs:
        index = info["Inputs"][input]["index"]
        dict_to_plot[input] = DataToVisualize(x[index], input, extent_highs)

    return dict_to_plot

def plot_datafields(data: Dict[str, DataToVisualize], name_pic: str, settings_pic: dict, only_inner: bool = False, plot_all_in_1_pic: bool = True):
    # plot datafields (temperature true, temperature out, error, physical variables (inputs))

    if plot_all_in_1_pic:
        num_subplots = len(data)
        fig, axes = plt.subplots(num_subplots, 1, sharex=True)
        fig.set_figheight(num_subplots*5)
        
        for index, (name, datapoint) in enumerate(data.items()):
            plt.sca(axes[index])
            plt.title(datapoint.name)
            if only_inner:
                plt.imshow(datapoint.data[100:400,100:400].T, **datapoint.imshowargs)
            else:  
                plt.imshow(datapoint.data.T, **datapoint.imshowargs)
            plt.gca().invert_yaxis()

            plt.ylabel("x [m]")
            _aligned_colorbar()

        plt.sca(axes[-1])
        plt.xlabel("y [m]")
        plt.tight_layout()
        ext_inner = "_inner" if only_inner else ""
        plt.savefig(f"{name_pic}{ext_inner}.{settings_pic['format']}", **settings_pic)
    else:
        for (name, datapoint) in data.items():
            fig, _ = plt.subplots(1, 1, sharex=True)
            fig.set_figheight(5)
            plt.title(datapoint.name)
            if only_inner:
                plt.imshow(datapoint.data[100:400,100:400].T, **datapoint.imshowargs)
            else:  
                plt.imshow(datapoint.data.T, **datapoint.imshowargs)
            plt.gca().invert_yaxis()

            plt.ylabel("x [m]")
            _aligned_colorbar()

            plt.xlabel("y [m]")
            plt.tight_layout()
            ext_inner = "_inner" if only_inner else ""
            plt.savefig(f"{name_pic}{ext_inner}_{name}.{settings_pic['format']}", **settings_pic)

def plot_isolines(data: Dict[str, DataToVisualize], name_pic: str, settings_pic: dict):
    # plot isolines of temperature fields
    num_subplots = 3 if "Original Temperature [C]" in data.keys() else 2
    fig, axes = plt.subplots(num_subplots, 1, sharex=True)
    fig.set_figheight(num_subplots)

    for index, name in enumerate(["t_true", "t_out", "Original Temperature [C]"]):
        try:
            plt.sca(axes[index])
            data[name].data = torch.flip(data[name].data, dims=[1])
            plt.title("Isolines of "+data[name].name)
            plt.contourf(data[name].data.T, **data[name].contourfargs)
            plt.ylabel("x [m]")
            _aligned_colorbar(ticks=[11.6, 15.6])
        except:
            pass

    plt.sca(axes[-1])
    plt.xlabel("y [m]")
    plt.tight_layout()
    plt.savefig(f"{name_pic}_isolines.{settings_pic['format']}", **settings_pic)

def infer_all_and_summed_pic(model: UNet, dataloader: DataLoader, device: str):
    '''
    sum inference time (including reverse-norming) and pixelwise error over all datapoints
    '''
    
    norm = dataloader.dataset.dataset.norm
    model.eval()

    current_id = 0
    avg_inference_time = 0
    summed_error_pic = torch.zeros_like(torch.Tensor(dataloader.dataset[0][0][0])).cpu()

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
            x = norm.reverse(x.cpu().detach().squeeze(), "Inputs")
            y = norm.reverse(y.cpu().detach(),"Labels")[0]
            y_out = norm.reverse(y_out.cpu().detach()[0],"Labels")[0]
            avg_inference_time += (time.perf_counter() - start_time)
            summed_error_pic += abs(y-y_out)

            current_id += 1

    avg_inference_time /= current_id
    summed_error_pic /= current_id
    return avg_inference_time, summed_error_pic

def plot_avg_error_cellwise(dataloader, summed_error_pic, settings_pic: dict):
    # plot avg error cellwise AND return time measurements for inference

    info = dataloader.dataset.dataset.info
    extent_highs = (np.array(info["CellsSize"][:2]) * dataloader.dataset[0][0][0].shape)
    extent = (0,int(extent_highs[0]),int(extent_highs[1]),0)

    fig = plt.figure()
    fig.set_figheight(5)
    plt.imshow(summed_error_pic.T, cmap="RdBu_r", extent=extent)
    plt.gca().invert_yaxis()
    plt.ylabel("x [m]")
    plt.xlabel("y [m]")
    plt.title("Cellwise averaged error [°C]")
    _aligned_colorbar()

    plt.tight_layout()
    plt.savefig(f"{settings_pic['folder']}/avg_error.{settings_pic['format']}", format=settings_pic['format'])

def _aligned_colorbar(*args, **kwargs):
    cax = make_axes_locatable(plt.gca()).append_axes(
        "right", size=0.3, pad=0.05)
    plt.colorbar(*args, cax=cax, **kwargs)
