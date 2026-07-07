from dataclasses import dataclass, field
from math import inf
from typing import Dict
# set backend for matplotlib, necessary if no GUI is available, e.g. in tmux
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from preprocessing.transforms import NormalizeTransform
from processing.networks.unetVariants import UNet
import postprocessing.cmap_jp

@dataclass
class DataToVisualize:
    data: np.ndarray
    category: str
    physical_property: str
    imshowargs: Dict = field(default_factory=dict)
    vmax: float = None
    vmin: float = None

    def __post_init__(self):
        self.imshowargs = {"cmap": "RdBu_r", 
                           "interpolation": "nearest",
                           "origin": "lower",}
        if self.vmax is not None:
            self.imshowargs["vmax"] = self.vmax
        if self.vmin is not None:
            self.imshowargs["vmin"] = self.vmin

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
    
    current_id = 0
    for inputs, labels in dataloader:
        len_batch = inputs.shape[0]
        for datapoint_id in range(len_batch):
            name_pic = f"{plot_path}_{current_id}"

            x = inputs[datapoint_id]
            y = labels[datapoint_id]

            y_out = model.infer(x.unsqueeze(0), args["device"])

            x, y, y_out = reverse_norm_one_dp(x, y, y_out, norm)
            dict_to_plot = prepare_data_to_plot(x, y, y_out, info)

            plot_datafields(dict_to_plot, name_pic, settings_pic)

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

    outs_max = [max(y_reduced[i].max(), y_out[i].max()) for i in range(len(y_reduced))]
    outs_min = [min(y_reduced[i].min(), y_out[i].min()) for i in range(len(y_reduced))]
    
    dict_to_plot = {}
    labels = info["Labels"].keys()
    for label in labels:
        index = info["Labels"][label]["index"]
        dict_to_plot[f"{label}_true"] = DataToVisualize(y_reduced[index], "Label", label, vmax=outs_max[index], vmin=outs_min[index])
        dict_to_plot[f"{label}_out"] = DataToVisualize(y_out[index], "Prediction", label, vmax=outs_max[index], vmin=outs_min[index])
        dict_to_plot[f"{label}_error"] = DataToVisualize(torch.abs(y_reduced[index]-y_out[index]), "Absolute Error", label)
    inputs = info["Inputs"].keys()
    for input in inputs:
        index = info["Inputs"][input]["index"]
        if "[" in input: # if input name contains units, remove them for the plot title
            input = input.split("[")[0]
        dict_to_plot[input] = DataToVisualize(x[index], "Input", input, (np.array(info["CellsSize"][:2]) * x.shape[-2:]))

    return dict_to_plot

def plot_datafields(data: Dict[str, DataToVisualize], name_pic: str, settings_pic: dict):
    # plot datafields (temperature true, temperature out, error, physical variables (inputs))
    for (name, datapoint) in data.items():
        fig, _ = plt.subplots(1, 1, sharex=True)
        fig.set_figheight(5)
        plt.title(datapoint.category)
        plt.imshow(datapoint.data, **datapoint.imshowargs)

        plt.ylabel("y [cells]")
        plt.xlabel("x [cells]")
        aligned_colorbar(label=datapoint.physical_property)
        plt.tight_layout()
        plt.savefig(f"{name_pic}_{name}.{settings_pic['format']}", **settings_pic)

def aligned_colorbar(*args, **kwargs):
    cax = make_axes_locatable(plt.gca()).append_axes(
        "right", size=0.3, pad=0.05)
    plt.colorbar(*args, cax=cax, **kwargs)

def interim_visu(model, dataloader, path_desti, device):
    with torch.no_grad():
        for inputs, labels in dataloader:
            len_batch = inputs.shape[0]
            # for dp_in, dp_lab in zip(inputs, labels):
            y_outs = model.infer(inputs.to(device), device)

            plt.figure(figsize=(12,12))
            for i in range(len_batch):
                plt.subplot(len_batch, 3, (i + 1) * 3 - 2)
                plt.imshow(labels[i,0].cpu().numpy(), origin="lower", cmap="RdBu_r")#,vmin=0,vmax=1)
                plt.title(f"Label {i}")
                aligned_colorbar()

                plt.subplot(len_batch, 3, (i + 1) * 3 - 1)
                plt.imshow(y_outs[i,0].cpu().detach().numpy(), origin="lower", cmap="RdBu_r")#,vmin=0,vmax=1)
                plt.title(f"Prediction {i}")
                aligned_colorbar()

                plt.subplot(len_batch, 3, (i + 1) * 3 - 0)
                plt.imshow(torch.abs(labels[i,0].cpu() - y_outs[i,0].cpu().detach()), origin="lower", cmap="RdBu_r")#,vmin=0,vmax=1)
                plt.title(f"Error {i}")
                aligned_colorbar()
            plt.tight_layout()
            plt.savefig(path_desti, dpi=300)
            break