from code.postprocessing.cmap_jp import *  # noqa: F403
from code.preprocessing.datasets.dataset import DatasetType
from code.preprocessing.transforms import NormalizeTransform
from code.processing.networks.unetVariants import UNet
from code.utils import logging as log  # noqa: F401
from copy import deepcopy
from dataclasses import dataclass, field
from math import inf

import matplotlib
import numpy as np
import torch
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

matplotlib.use("Agg")


@dataclass
class DataToVisualize:
    datasetType: DatasetType
    data: np.ndarray
    category: str
    physical_property: str
    extent_highs: tuple[float, float] = (1280, 100)
    imshowargs: dict = field(default_factory=dict)
    vmax: float | None = None
    vmin: float | None = None
    dark_mode: bool = False
    temperature_spread = 5
    ambient_temp = 10.6

    def __post_init__(self):
        extent = (0, int(self.extent_highs[0]), int(self.extent_highs[1]), 0)

        match self.physical_property:
            case (
                "Liquid X-Velocity [m_per_y]"
                | "Liquid Y-Velocity [m_per_y]"
                | "Liquid Z-Velocity [m_per_y]"
                | "Liquid Pressure [Pa]"
                | "Permeability X [m^2]"
                | "Pressure Gradient [-]"
            ):
                cmap = "jp_linear_dark" if self.dark_mode else "jp_linear"
            case (
                "Streamlines Faded [-]"
                | "Streamlines Faded Outer [-]"
                | "Streamline-Sum_Position"
                | "Streamline-Sum_RelativeUncertainty"
                | "Streamline-Sum_TimeFaded-Position"
                | "Streamline-Max_TimeFaded"
                | "Streamline-Max_TimeSeasons"
            ):
                self.vmin = 0
                self.vmax = 1
                cmap = "jp_linear_dark" if self.dark_mode else "jp_linear"
            case "Material ID":
                cmap = "binary"
                self.vmin = 0
                self.vmax = 1
            case "Line Integral Convolution":
                cmap = "bone"
            case "Temperature [C]" | "Streamline-TemperatureApproximation" | "Streamline-Sum_TimeSeasons-Position":
                match self.datasetType:
                    case DatasetType.seasonal:
                        cmap = "jp_temperature_bidirectional"
                        self.vmin = self.ambient_temp - self.temperature_spread
                        self.vmax = self.ambient_temp + self.temperature_spread
                    case DatasetType.steady_state_heating:
                        cmap = "jp_temperature_upperlinear"
                        self.vmin = self.ambient_temp
                        self.vmax = self.ambient_temp + self.temperature_spread
                    case DatasetType.steady_state_cooling:
                        cmap = "jp_temperature_lowerlinear"
                        self.vmin = self.ambient_temp - self.temperature_spread
                        self.vmax = self.ambient_temp
                    case _:
                        raise ValueError(f"Unknown dataset type: {self.datasetType}")
                if self.dark_mode:
                    cmap += "_dark"
                if self.physical_property in [
                    "Streamline-TemperatureApproximation",
                    "Streamline-Sum_TimeSeasons-Position",
                ]:
                    self.vmin = 0
                    self.vmax = 1
            case _:
                raise ValueError(f"Unknown physical property: {self.physical_property}")

        self.imshowargs = {
            "cmap": cmap,
            "extent": extent,
            "interpolation": "nearest",
        }

        if self.vmax is not None:
            self.imshowargs["vmax"] = self.vmax
        if self.vmin is not None:
            self.imshowargs["vmin"] = self.vmin

        self._normalize_labels()

    def _normalize_labels(self):
        mapping = {
            "Liquid Pressure [Pa]": "Pressure [Pa]",
            "Material ID": "Positions of Heat Pumps [-]",
            "Permeability X [m^2]": "Permeability [m$^2$]",
            "SDF": "SDF-Transformed Positions of Heat Pumps [-]",
            "MDF": "MDF-Transformed Positions of Heat Pumps [-]",
            "Streamlines Fade": "Streamlines Fade [-]",
            "Streamlines": "Streamlines [-]",
        }
        if self.physical_property in mapping:
            self.physical_property = mapping[self.physical_property]


def aligned_colorbar(ax, im, **kwargs):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.3, pad=0.05)
    cb = ax.figure.colorbar(im, cax=cax, **kwargs)
    cb.ax.tick_params(labelsize=20)


def plot_datapoint(
    name: str,
    datapoint: DataToVisualize,
    name_pic: str,
    settings_pic: dict,
    only_inner: bool = False,
    remove_axis: bool = False,
):
    is_streamline = name.startswith("Streamline")

    if remove_axis and is_streamline:
        fig = Figure(figsize=(8, 6.5))
        # Use a copy to avoid side effects on the settings dict
        settings = settings_pic.copy()
        settings["bbox_inches"] = "tight"
        settings["pad_inches"] = 0

        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
    else:
        fig = Figure(figsize=(8.4, 6.5))
        settings = settings_pic
        ax = fig.add_subplot(1, 1, 1)
        settings["bbox_inches"] = "tight"
        ax.tick_params(axis="both", which="major", labelsize=20)

    imshow_args = datapoint.imshowargs.copy()

    if "error" in name_pic.lower():
        imshow_args["cmap"] = "jp_linear"
        imshow_args["vmax"] = None
        imshow_args["vmin"] = None

    data_to_show = datapoint.data[100:400, 100:400] if only_inner else datapoint.data
    im = ax.imshow(data_to_show, **imshow_args)

    if "error" not in name_pic.lower() and datapoint.vmax is not None and datapoint.vmin is not None:
        im.set_clim(datapoint.vmin, datapoint.vmax)

    if not (remove_axis and is_streamline):
        aligned_colorbar(ax, im)
        fig.set_tight_layout(True)

    ext_inner = "_inner" if only_inner else ""
    fig.savefig(f"{name_pic}{ext_inner}.{settings['format']}", **settings)


def plot_datafields(
    data: dict[str, DataToVisualize],
    name_pic: str,
    settings_pic: dict,
    only_inner: bool = False,
    plot_all_in_1_pic: bool = True,
):
    if plot_all_in_1_pic:
        num_subplots = len(data)
        fig = Figure(figsize=(8.4, num_subplots * 4))

        axes = fig.subplots(num_subplots, 1, sharex=True)
        if num_subplots == 1:
            axes = [axes]

        for index, (_, datapoint) in enumerate(data.items()):
            ax = axes[index]
            ax.set_title(datapoint.category, fontsize=16)

            data_to_show = datapoint.data[100:400, 100:400] if only_inner else datapoint.data
            ax.imshow(data_to_show, **datapoint.imshowargs)

            ax.invert_yaxis()
            ax.set_ylabel("x [m]", fontsize=20)
            ax.tick_params(axis="both", which="major", labelsize=20)

        axes[-1].set_xlabel("y [m]", fontsize=20)
        fig.set_tight_layout(True)

        ext_inner = "_inner" if only_inner else ""
        fig.savefig(f"{name_pic}{ext_inner}.{settings_pic['format']}", **settings_pic)
    else:
        for name, datapoint in data.items():
            log.info(f"Plotting {name}...")
            plot_datapoint(name, datapoint, f"{name_pic}_{name}", settings_pic, only_inner)


def prepare_data_to_plot_sequential(
    datasetType: DatasetType, x: torch.Tensor, y: torch.Tensor, y_out: torch.Tensor, info: dict
):
    # prepare data of temperature true, temperature out, error, physical variables (inputs)
    required_size = y_out.shape[-2:]
    log.info(f"Required size: {required_size}")
    # start_pos = ((y.shape[1] - required_size[1])//2, (y.shape[2] - required_size[2])//2)
    # y_reduced = y[:,start_pos[0]:start_pos[0]+required_size[1], start_pos[1]:start_pos[1]+required_size[2]]
    y_reduced = y.squeeze_()
    y_out = y_out.squeeze_()
    outs_max = [max(y_reduced.max(), y_out.max()) for idx in range(len(y_reduced))]
    outs_min = [min(y_reduced.min(), y_out.min()) for idx in range(len(y_reduced))]
    extent_highs_y = np.array(info["CellsSize"][:2]) * y_out.shape[-2:]

    dict_to_plot = {}
    labels = info["Labels"].keys()

    for label in labels:
        index = info["Labels"][label]["index"]
        for time_step in range(y_reduced.shape[0]):
            dict_to_plot[f"{label}_true at time {time_step}"] = DataToVisualize(
                datasetType,
                y_reduced[time_step],
                "Label",
                label,
                extent_highs_y,
                vmax=outs_max[index],
                vmin=outs_min[index],
            )
            dict_to_plot[f"{label}_out at time {time_step}"] = DataToVisualize(
                datasetType,
                y_out[time_step],
                "Prediction",
                label,
                extent_highs_y,
                vmax=outs_max[index],
                vmin=outs_min[index],
            )
            dict_to_plot[f"{label}_error at time {time_step}"] = DataToVisualize(
                datasetType, torch.abs(y_reduced[time_step] - y_out[time_step]), "Absolute Error", label, extent_highs_y
            )
    inputs = info["Inputs"].keys()
    for input in inputs:
        index = info["Inputs"][input]["index"]
        dict_to_plot[input] = DataToVisualize(
            datasetType, x[index].squeeze_(), "Input", input, (np.array(info["CellsSize"][:2]) * x.shape[-2:])
        )

    return dict_to_plot


def reverse_norm_one_dp_sequence(x: torch.Tensor, y: torch.Tensor, y_out: torch.Tensor, norm: NormalizeTransform):
    # reverse transform for plotting real values
    x = norm.reverse(x.detach().cpu(), "Inputs")
    if len(y.shape) == 4:
        y = norm.reverse(y.detach().cpu(), "Labels")
    else:
        y = norm.reverse(y.detach().cpu(), "Labels")
    try:
        y_out = norm.reverse(y_out.detach().cpu().squeeze(0), "Labels")
    except Exception:
        y_out = norm.reverse(y_out.squeeze(0), "Labels")
    return x, y, y_out


def reverse_norm_one_dp_outputs(y_out: torch.Tensor, norm: NormalizeTransform):
    y_sq = y_out.detach().cpu().squeeze(0)
    try:
        return norm.reverse(y_sq, "Labels")
    except TypeError:
        return norm.reverse(y_out.squeeze(0), "Labels")


def reverse_norm_one_dp_inputs(x: torch.Tensor, y: torch.Tensor, norm: NormalizeTransform):
    x_rev = norm.reverse(x.detach().cpu().squeeze(0), "Inputs")
    y_detached = y.detach().cpu()
    if len(y.shape) == 4:
        y_rev = norm.reverse(y_detached.squeeze(0), "Labels")
    else:
        y_rev = norm.reverse(y_detached, "Labels")
    return x_rev, y_rev


def prepare_data_to_plot_inputs(
    datasetType: DatasetType, x: torch.Tensor, y: torch.Tensor, info: dict
) -> dict[str, DataToVisualize]:
    required_size = y.shape
    h_diff = y.shape[1] - required_size[1]
    w_diff = y.shape[2] - required_size[2]

    start_h = h_diff // 2
    start_w = w_diff // 2
    y_reduced = y[:, start_h : start_h + required_size[1], start_w : start_w + required_size[2]]

    num_channels = len(y_reduced)
    outs_max = [y_reduced[i].max().item() for i in range(num_channels)]
    outs_min = [y_reduced[i].min().item() for i in range(num_channels)]

    dict_to_plot = {}
    for input_name, input_info in info["Inputs"].items():
        idx = input_info["index"]
        extent_vals = np.array(info["CellsSize"][:2]) * x.shape[-2:]
        dict_to_plot[input_name] = DataToVisualize(
            datasetType=datasetType,
            data=x[idx],
            category="",
            physical_property=input_name,
            extent_highs=tuple(extent_vals),
        )
    for label_name, label_info in info["Labels"].items():
        idx = label_info["index"]
        extent_vals = np.array(info["CellsSize"][:2]) * y.shape[-2:]
        dict_to_plot[f"{label_name}_true"] = DataToVisualize(
            datasetType=datasetType,
            data=y_reduced[idx],
            category="Label",
            physical_property=label_name,
            extent_highs=tuple(extent_vals),
            vmax=outs_max[idx],
            vmin=outs_min[idx],
        )

    return dict_to_plot


def prepare_data_to_plot_outputs(
    datasetType: DatasetType, y: torch.Tensor, y_out: torch.Tensor, info: dict, lic: bool = False
) -> dict[str, DataToVisualize]:
    required_size = y_out.shape
    h_diff = y.shape[1] - required_size[1]
    w_diff = y.shape[2] - required_size[2]

    start_h = h_diff // 2
    start_w = w_diff // 2
    y_reduced = y[:, start_h : start_h + required_size[1], start_w : start_w + required_size[2]]

    num_channels = len(y_reduced)
    outs_max = [y_reduced[i].max().item() for i in range(num_channels)]
    outs_min = [y_reduced[i].min().item() for i in range(num_channels)]

    extent_vals = np.array(info["CellsSize"][:2]) * y_out.shape[-2:]
    dict_to_plot = {}

    if lic:
        import lic

        index = info["Labels"]["Liquid X-Velocity [m_per_y]"]["index"]
        temp_x = y_reduced[index].cpu().numpy()  # Auf CPU/NumPy konvertieren
        index = info["Labels"]["Liquid Y-Velocity [m_per_y]"]["index"]
        temp_y = y_reduced[index].cpu().numpy()  # Auf CPU/NumPy konvertieren

        lic_result = lic.lic(temp_y, temp_x, length=30)
        dict_to_plot["LIC"] = DataToVisualize(
            lic_result, "LIC", "LIC", extent_vals, vmax=np.max(lic_result), vmin=np.min(lic_result)
        )

    for label_name, label_info in info["Labels"].items():
        idx = label_info["index"]
        dict_to_plot[f"{label_name}_out"] = DataToVisualize(
            datasetType=datasetType,
            data=y_out[idx],
            category="Prediction",
            physical_property=label_name,
            extent_highs=tuple(extent_vals),
            vmax=outs_max[idx],
            vmin=outs_min[idx],
        )
        dict_to_plot[f"{label_name}_error"] = DataToVisualize(
            datasetType=datasetType,
            data=torch.abs(y_reduced[idx] - y_out[idx]),
            category="Absolute Error",
            physical_property=label_name,
            extent_highs=tuple(extent_vals),
        )

    return dict_to_plot


def visualize_inputs(
    datasetType: DatasetType,
    dataloader,
    args: dict,
    amount_datapoints_to_visu: int = inf,
    plot_path: str = "default",
    pic_format: str = "png",
):
    log.info("Visualizing Inputs...")
    if dataloader.dataset.__class__.__name__ == "SimulationDatasetCutsSequential":
        log.info("skipping ...")
        return

    total_samples = len(dataloader.dataset)
    limit = min(amount_datapoints_to_visu, total_samples)

    try:
        norm = dataloader.dataset.norm
        info = dataloader.dataset.info
    except AttributeError:
        norm = dataloader.dataset.dataset.norm
        info = dataloader.dataset.dataset.info

    settings_pic = {"format": pic_format, "dpi": 160}
    current_count = 0

    for inputs, labels in dataloader:
        log.info(inputs.shape, labels.shape, "shape of inputs and labels")
        batch_size = inputs.shape[0]

        for i in range(batch_size):
            if current_count >= limit:
                return

            name_pic = f"{plot_path}_{current_count}_input"
            x, y = reverse_norm_one_dp_inputs(inputs[i], labels[i], norm)

            dict_to_plot = prepare_data_to_plot_inputs(datasetType, x, y, info)
            plot_datafields(dict_to_plot, name_pic, settings_pic, only_inner=False, plot_all_in_1_pic=False)

            current_count += 1


def visualize_outputs(
    datasetType: DatasetType,
    model: UNet,
    dataloader,
    args: dict,
    amount_datapoints_to_visu: int = inf,
    plot_path: str = "default",
    pic_format: str = "png",
    scaleBounds=None,
    useNonLinearCmap: bool = None,
):
    log.info("Visualizing Outputs...")
    total_samples = len(dataloader.dataset)
    limit = min(amount_datapoints_to_visu, total_samples)

    try:
        norm = dataloader.dataset.norm
        info = dataloader.dataset.info
    except AttributeError:
        norm = dataloader.dataset.dataset.norm
        info = dataloader.dataset.dataset.info

    settings_pic = {"format": pic_format, "dpi": 160}
    current_count = 0
    device = args["device"]

    for inputs, labels in dataloader:
        log.info(inputs.shape, labels.shape, "shape of inputs and labels")
        batch_size = inputs.shape[0]

        for i in range(batch_size):
            if current_count >= limit:
                return

            name_pic = f"{plot_path}_{current_count}_output"
            x = inputs[i]
            y = labels[i]
            if dataloader.dataset.__class__.__name__ == "SimulationDatasetCutsSequential":
                y_out = model.infer(x.unsqueeze(0), args["device"])

                x = x[:-1]
                x, y, y_out = reverse_norm_one_dp_sequence(x, y, y_out, norm)
                dict_to_plot = prepare_data_to_plot_sequential(datasetType, x, y, y_out, info)
            else:
                # deepcopy to avoid in-place operations messing up gradients
                x_copy = deepcopy(x)
                y_copy = deepcopy(y)

                y_out_raw = model.infer(x_copy.unsqueeze(0), device)

                y_out = reverse_norm_one_dp_outputs(y_out_raw, norm)
                _, y_denorm = reverse_norm_one_dp_inputs(x_copy, y_copy, norm)

                dict_to_plot = prepare_data_to_plot_outputs(datasetType, y_denorm, y_out, info, lic=False)

            plot_datafields(dict_to_plot, name_pic, settings_pic, only_inner=False, plot_all_in_1_pic=False)

            current_count += 1


def visualize_streamlines(
    datasetType: DatasetType, output_name: str, prop_name: str, tensor_data: torch.Tensor
) -> None:
    """Generates plots for all streamlines."""
    resolution = [12800, 12800]

    plot_datapoint(
        name=f"Streamlines - {prop_name}",
        datapoint=DataToVisualize(datasetType, tensor_data, "Input", prop_name, resolution),
        name_pic=str(output_name),
        settings_pic={"format": "png", "dpi": 160},
        only_inner=False,
        remove_axis=False,
    )
