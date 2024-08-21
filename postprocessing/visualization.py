import time
from dataclasses import dataclass, field
from math import inf
from typing import Dict
import pathlib

import matplotlib as mpl
mpl.use('pdf') #pgf')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from preprocessing.transforms import NormalizeTransform
from processing.networks.unet import UNet
from processing.pipelines.extend_plumes_old import infer_nopad, update_params
from postprocessing.visu_utils import _aligned_colorbar

@dataclass
class DataToVisualize:
    data: np.ndarray
    category: str
    physical_property: str
    extent_highs :tuple = (1280,100) # x,y in meters
    imshowargs: Dict = field(default_factory=dict)
    contourfargs: Dict = field(default_factory=dict)
    contourargs: Dict = field(default_factory=dict)
    vmax: float = None
    vmin: float = None

    def __post_init__(self):
        extent = (0,int(self.extent_highs[0]),int(self.extent_highs[1]),0)

        self.imshowargs = {"cmap": "RdBu_r", 
                           "extent": extent,
                           "interpolation": "nearest",}
        if self.vmax is not None:
            self.imshowargs["vmax"] = self.vmax
        if self.vmin is not None:
            self.imshowargs["vmin"] = self.vmin

        self.contourfargs = {"levels": np.arange(10.4, 16, 0.25), 
                             "cmap": "RdBu_r", 
                             "extent": extent}
        
        T_gwf = 10.6
        T_inj_diff = 5.0
        self.contourargs = {"levels" : [np.round(T_gwf + 1, 1)],
                            "cmap" : "Pastel1", 
                            "extent": extent}

        if self.physical_property == "Liquid Pressure [Pa]":
            self.physical_property = "Pressure [Pa]"
        elif self.physical_property == "Material ID":
            self.physical_property = "Positions of Heat Pumps [-]"
        elif self.physical_property == "Permeability X [m^2]":
            self.physical_property = "Permeability [m$^2$]"
        elif self.physical_property == "SDF":
            self.physical_property = "SDF-Transformed Positions of Heat Pumps [-]"
        elif self.physical_property == "MDF":
            self.physical_property = "MDF-Transformed Positions of Heat Pumps [-]"
        elif self.physical_property == "Streamlines Fade":
            self.physical_property = "Streamlines Fade [-]"
        elif self.physical_property == "Streamlines":
            self.physical_property = "Streamlines [-]"

def visualizations(model: UNet, dataloader: DataLoader, args: dict, amount_datapoints_to_visu: int = inf, plot_path: str = "default", pic_format: str = "png"):
    print("Visualizing...") #, end="\r")

    if amount_datapoints_to_visu > len(dataloader.dataset):
        amount_datapoints_to_visu = len(dataloader.dataset)

    try:
        norm = dataloader.dataset.norm
        info = dataloader.dataset.info
    except AttributeError:
        norm = dataloader.dataset.dataset.norm
        info = dataloader.dataset.dataset.info
    settings_pic = {"format": pic_format,
                    "dpi": 1200,}
    
    if args["problem"] == "extend":
        params = {"start_visu" : 0,
                    "end_visu" : 1000,
                    "start_input_box" : 64,
                    "skip_in_field" : 32,
                    "rm_boundary_l" : 16,
                    "rm_boundary_r" : int(16/2),}
        params = update_params(params, args["model"], temp_norm = norm)

    current_id = 0
    for inputs, labels in dataloader:
        print(inputs.shape, labels.shape, "shape of inputs and labels")
        len_batch = inputs.shape[0]
        for datapoint_id in range(len_batch):
            name_pic = f"{plot_path}_{current_id}"

            x = inputs[datapoint_id] #.to(args["device"])
            y = labels[datapoint_id] #.to(args["device"])

            if args["problem"] == "extend":
                y_out = infer_nopad(model, x, y, params, overlap=True, device=args["device"])
            else:
                y_out = model.infer(x.unsqueeze(0), args["device"])

            x, y, y_out = reverse_norm_one_dp(x, y, y_out, norm)
            dict_to_plot = prepare_data_to_plot(x, y, y_out, info) # TODO vorher: y[0], y_out[0]

            if args["problem"]== "allin1":
                # if settings.case=="test":
                #     plot_datafields(dict_to_plot, name_pic, settings_pic, only_inner=True)
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
    if len(y.shape) == 4:
        y = norm.reverse(y.detach().cpu().squeeze(0),"Labels")
    else:
        y = norm.reverse(y.detach().cpu(),"Labels")
    try:
        y_out = norm.reverse(y_out.detach().cpu().squeeze(0),"Labels")
    except:
        y_out = norm.reverse(y_out.squeeze(0),"Labels")
    return x, y, y_out

def prepare_data_to_plot(x: torch.Tensor, y: torch.Tensor, y_out:torch.Tensor, info: dict):
    # prepare data of temperature true, temperature out, error, physical variables (inputs)
    required_size = y_out.shape
    start_pos = ((y.shape[1] - required_size[1])//2, (y.shape[2] - required_size[2])//2)
    y_reduced = y[:,start_pos[0]:start_pos[0]+required_size[1], start_pos[1]:start_pos[1]+required_size[2]]
    print("required", required_size, "start", start_pos, "y shape", y.shape, "y_reduced shape", y_reduced.shape)
    
    outs_max = [max(y_reduced[idx].max(), y_out[idx].max()) for idx in range(len(y_reduced))]
    outs_min = [min(y_reduced[idx].min(), y_out[idx].min()) for idx in range(len(y_reduced))]
    extent_highs = (np.array(info["CellsSize"][:2]) * x.shape[-2:])

    dict_to_plot = {}
    labels = info["Labels"].keys()
    for label in labels:
        index = info["Labels"][label]["index"]
        print(outs_max[index], outs_min[index])
        dict_to_plot[f"{label}_true"] = DataToVisualize(y_reduced[index], "Label", label, extent_highs, vmax=outs_max[index], vmin=outs_min[index])
        dict_to_plot[f"{label}_out"] = DataToVisualize(y_out[index], "Prediction", label, extent_highs, vmax=outs_max[index], vmin=outs_min[index])
        dict_to_plot[f"{label}_error"] = DataToVisualize(torch.abs(y_reduced[index]-y_out[index]), "Absolute Error", label, extent_highs)
    # inputs = info["Inputs"].keys()
    # for input in inputs:
    #     index = info["Inputs"][input]["index"]
    #     dict_to_plot[input] = DataToVisualize(x[index], "Input", input, extent_highs)

    return dict_to_plot

def plot_datafields(data: Dict[str, DataToVisualize], name_pic: str, settings_pic: dict, only_inner: bool = False, plot_all_in_1_pic: bool = True):
    # plot datafields (temperature true, temperature out, error, physical variables (inputs))

    if plot_all_in_1_pic:
        num_subplots = len(data)
        fig, axes = plt.subplots(num_subplots, 1, sharex=True)
        fig.set_figheight(num_subplots*3)
        
        for index, (name, datapoint) in enumerate(data.items()):
            plt.sca(axes[index])
            plt.title(datapoint.category)
            if only_inner:
                plt.imshow(datapoint.data[100:400,100:400].T, **datapoint.imshowargs)
            else:  
                plt.imshow(datapoint.data.T, **datapoint.imshowargs)
            plt.gca().invert_yaxis()

            plt.ylabel("x [m]")
            _aligned_colorbar(label=datapoint.physical_property)

        # plt.sca(axes[-1]) # TODO DONT WANT THAT ANYMORE??? FLIPS MY PICS
        plt.xlabel("y [m]")
        plt.tight_layout()
        ext_inner = "_inner" if only_inner else ""
        plt.savefig(f"{name_pic}{ext_inner}.{settings_pic['format']}", **settings_pic)
    else:
        for (name, datapoint) in data.items():
            fig, _ = plt.subplots(1, 1, sharex=True)
            fig.set_figheight(5)
            plt.title(datapoint.category)
            if only_inner:
                plt.imshow(datapoint.data[100:400,100:400].T, **datapoint.imshowargs)
            else:  
                plt.imshow(datapoint.data.T, **datapoint.imshowargs)
            # plt.gca().invert_yaxis() # TODO DONT WANT THAT ANYMORE!! FLIPS MY PICS

            plt.ylabel("x [m]")
            plt.xlabel("y [m]")
            _aligned_colorbar(label=datapoint.physical_property)
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
            plt.title("Isolines of "+data[name].category)
            plt.contourf(data[name].data.T, **data[name].contourfargs)
            plt.ylabel("x [m]")
            _aligned_colorbar(label=data[name].physical_property, ticks=[11.6, 15.6])
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
    plt.title("Cellwise Averaged Error")
    _aligned_colorbar(label="Temperature [°C]")

    plt.tight_layout()
    plt.savefig(f"{settings_pic['folder']}/avg_error.{settings_pic['format']}", format=settings_pic['format'])
