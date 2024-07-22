import time
from dataclasses import dataclass, field
from math import inf
from typing import Dict

# import matplotlib as mpl
# mpl.use('pgf')
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.utils.data import DataLoader

from data_stuff.transforms import NormalizeTransform
from networks.unet import UNet

# mpl.rcParams.update({'figure.max_open_warning': 0})
# plt.rcParams['figure.figsize'] = [16, 5]

# TODO: look at vispy library for plotting 3D data

@dataclass
class DataToVisualize:
    data: np.ndarray
    name: str
    extent_highs :tuple #= (1280,100) # x,y in meters
    imshowargs: Dict = field(default_factory=dict)
    contourfargs: Dict = field(default_factory=dict)
    contourargs: Dict = field(default_factory=dict)
    x_lim = (0,640)
    y_lim = (0,64)
    cmap: str = "hot"

    def __post_init__(self):
        #extent = (,int(self.extent_highs[0]),int(self.extent_highs[1]),0)

        self.imshowargs.update({"cmap": self.cmap, 
                           "extent": self.extent_highs})

        self.contourfargs = {"levels": np.arange(10.4, 16, 0.25), 
                             "cmap": "terrain", 
                             "extent": self.extent_highs}
        
        T_gwf = 10.6
        T_inj_diff = 5.0
        self.contourargs = {"levels" : [np.round(T_gwf + 1, 1)],
                            "cmap" : "Pastel1", 
                            "extent": self.extent_highs}

        if self.name == "Liquid Pressure [Pa]":
            self.name = "Pressure in [Pa]"
        elif self.name == "Material ID":
            self.name = "Position of the heatpump in [-]"
        elif self.name == "Permeability X [m^2]":
            self.name = "Permeability in [m$^2$]"
        elif self.name == "SDF":
            self.name = "SDF-transformed position in [-]"
    
def visualizations(model: UNet, dataloader: DataLoader, device: str, amount_datapoints_to_visu: int = inf, plot_path: str = "default", pic_format: str = "png"):
    print("Visualizing...", end="\r")

    if amount_datapoints_to_visu > len(dataloader.dataset):
        amount_datapoints_to_visu = len(dataloader.dataset)

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

            x = torch.unsqueeze(inputs[datapoint_id].to(device), 0)
            y = labels[datapoint_id]
            y_out = model(x).to(device)

            x, y, y_out = reverse_norm_one_dp(x, y, y_out, norm)
            dict_to_plot = prepare_data_to_plot(x, y, y_out, info)

            plot_datafields(dict_to_plot, name_pic, settings_pic)
            # plot_isolines(dict_to_plot, name_pic, settings_pic)
            # measure_len_width_1K_isoline(dict_to_plot)

            if current_id >= amount_datapoints_to_visu-1:
                return None
            current_id += 1

def visualizations_convLSTM(model: UNet, dataloader: DataLoader, device: str, last_cell_mode: str, dp_to_visu: np.array = inf, plot_path: str = "default", pic_format: str = "png"):
    print("Visualizing...", end="\r")

    norm = dataloader.dataset.dataset.norm
    info = dataloader.dataset.dataset.info
    model.eval()
    settings_pic = {"format": pic_format,
                    "dpi": 800,}

    current_id = 0
    batch_id = 1
    printed = 0
    print(f'dp_to_visu: {dp_to_visu}')
    for batch_id, (inputs, labels) in enumerate(dataloader, start=0):
        for dp_in_batch in range(dataloader.batch_size):
            datapoint_id = batch_id * dataloader.batch_size + dp_in_batch
            if datapoint_id in dp_to_visu:
                name_pic = f"{plot_path}/dp_{datapoint_id}"
                x = torch.unsqueeze(inputs[dp_in_batch].to(device), 0)
                y = labels[dp_in_batch]
                y_out = model(x).to(device)

                x, y, y_out = reverse_norm_one_dp(x, y, y_out, norm)
                
                dict_to_plot = prepare_data_to_plot_convLSTM(x, y.squeeze(), y_out.squeeze(), info, last_cell_mode)

                plot_datafields(dict_to_plot, name_pic, settings_pic)
                printed += 1

            if printed >= len(dp_to_visu):
                return None
            
            

def reverse_norm_one_dp(x: torch.Tensor, y: torch.Tensor, y_out:torch.Tensor, norm: NormalizeTransform):
    # reverse transform for plotting real values
    #x_split = torch.split(x, 1, 1)
    #x = torch.cat(x_split[:4], dim=1)
    x1 = norm.reverse(x[0][:4].detach().cpu(), "Inputs")
    x2 = norm.reverse(x[0][4].detach().cpu().unsqueeze(0), "Labels")
    x = torch.cat((x1,x2),dim=0)

    
    y = y.unsqueeze(0)
    y = norm.reverse(y.detach().cpu(), "Labels")

    y_out = norm.reverse(y_out.detach().cpu(),"Labels")

    return x, y, y_out

def prepare_data_to_plot_convLSTM(x: torch.Tensor, y: torch.Tensor, y_out:torch.Tensor, info: dict, last_cell_mode:str):
    # prepare data of temperature true, temperature out, error, physical variables (inputs)
    # Shape of x: torch.Size([4, 10, 64, 64])
    # Shape of y: torch.Size([10, 64, 64])
    # Shape of y_out: torch.Size([64, 64])
    
    perm = x[1]
    perm = perm.reshape(640,64)
    press = x[0]
    press = press.reshape(640,64)
    if last_cell_mode in ["none", "perm"]:
        temp = x[4][:-1]
        print(f'shape of temp: {temp.shape}')
        temp = temp.reshape(576,64)
        extent = (0,576,64,0)
    else:
        temp = x[4]
        temp = temp.reshape(640,64)
        extent = (0,640,64,0)
    #temp = temp[:576]
    press_max = info['Inputs']['Pressure Gradient [-]']['max']
    press_min = info['Inputs']['Pressure Gradient [-]']['min']
    #y = y.reshape(64,64)
    temp_max = max(max(y.max(), y_out.max()), temp.max())
    temp_min = min(min(y.min(), y_out.min()), temp.min())
    extent_highs = (0,640,64,0)
    #extent_highs = (np.array(info["CellsSize"][:2]) * x.shape[-2:])

    dict_to_plot = {     
        "sdf" : DataToVisualize(x[2].reshape(640,64), "Input: Signed Distance Function", (0,640,64,0), cmap="viridis"), 
        "press" : DataToVisualize(press, "Input: Pressure Gradient", (0,640,64,0), {"vmax": press_max, "vmin": press_min}, cmap="viridis"),
        "perm" : DataToVisualize(perm,  "Input: Permeabilität",(0,640,64,0), cmap="viridis"),
        "temp" : DataToVisualize(temp, "Input: Temperature in [°C]", extent,{"vmax": temp_max, "vmin": temp_min}),
        "t_true": DataToVisualize(y, f"Label: Temperature in [°C]", (576, 640, 64, 0),{"vmax": temp_max, "vmin": temp_min}),
        "t_out": DataToVisualize(y_out, "Prediction: Temperature in [°C]",(575,640,64,0), {"vmax": temp_max, "vmin": temp_min}),
        "error": DataToVisualize(torch.abs(y[-65:]-y_out), "Absolute error in [°C]",(575,640,64,0)),
    }

    return dict_to_plot

def prepare_data_to_plot(x: torch.Tensor, y: torch.Tensor, y_out:torch.Tensor, info: dict):
    # prepare data of temperature true, temperature out, error, physical variables (inputs)
    temp_max = max(y.max(), y_out.max())
    temp_min = min(y.min(), y_out.min())
    extent_highs = (np.array(info["CellsSize"][:2]) * x.shape[-2:])

    dict_to_plot = {
        "t_true": DataToVisualize(y, "Label: Temperature in [°C]", extent_highs, {"vmax": temp_max, "vmin": temp_min}),
        "t_out": DataToVisualize(y_out, "Prediction: Temperature in [°C]", extent_highs, {"vmax": temp_max, "vmin": temp_min}),
        "error": DataToVisualize(torch.abs(y-y_out), "Absolute error in [°C]", extent_highs),
    }
    inputs = info["Inputs"].keys()
    for input in inputs:
        index = info["Inputs"][input]["index"]
        dict_to_plot[input] = DataToVisualize(x[index], input, extent_highs)

    return dict_to_plot

def plot_datafields(data: Dict[str, DataToVisualize], name_pic: str, settings_pic: dict):
    # plot datafields (temperature true, temperature out, error, physical variables (inputs))

    num_subplots = len(data)
    fig, axes = plt.subplots(num_subplots, 1, sharex=True)
    fig.set_figheight(num_subplots)
    
    for index, (name, datapoint) in enumerate(data.items()):
        plt.sca(axes[index])
        axes[index].set_xlim(0,640)
        axes[index].set_ylim(0,64)
        plt.title(datapoint.name)
        plt.imshow(datapoint.data.T, **datapoint.imshowargs)
        #plt.gca().invert_yaxis()

        plt.ylabel("x [m]")
        _aligned_colorbar()

    plt.sca(axes[-1])
    plt.xlabel("y [m]")
    plt.tight_layout()
    plt.savefig(f"{name_pic}.{settings_pic['format']}", **settings_pic)

    
def plot_datafields_convLSTM(data: Dict[str, DataToVisualize], name_pic: str, settings_pic: dict):
    # plot datafields (temperature true, temperature out, error, physical variables (inputs))

    # num_subplots = len(data)
    # fig, axes = plt.subplots()
    # fig, axes = plt.subplots(num_subplots, 19, figsize=(30,10), sharex=True)
    # fig.set_figheight(num_subplots)
    
    # for row_index, (name, datapoint) in enumerate(data.items()):
    #     if len(datapoint.data.shape) > 2: 
    #         for column_index, datapoint_time_step in enumerate(datapoint.data):
    #             plt.sca(axes[row_index][column_index])
    #             plt.title(datapoint.name)
    #             plt.imshow(datapoint_time_step, **datapoint.imshowargs)
    #             plt.gca().invert_yaxis()
    #     else:
    #         plt.sca(axes[row_index][0])
    #         #plt.title(datapoint.name)
    #         plt.imshow(datapoint.data[0], **datapoint.imshowargs)
    #         plt.gca().invert_yaxis()

    #     plt.ylabel("x [m]")
    #     _aligned_colorbar()

    #     plt.ylabel("x [m]")
    #     _aligned_colorbar()

    #plt.sca(axes[-1])
    #plt.xlabel("y [m]")
    #plt.tight_layout()
    plt.savefig(f"{name_pic}.{settings_pic['format']}", **settings_pic)

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

    plt.figure()
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
