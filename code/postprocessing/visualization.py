from code.postprocessing.cmap_jp import *  # noqa: F403
from code.preprocessing.data_init import _label_has_midcell_above_threshold
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


FONT_SIZES = {
    "title": 13,
    "axis_label": 13,
    "tick_label": 13,
    "colorbar_label": 13,
    "colorbar_tick": 13,
    "suptitle": 13,
}

SAVEFIG_DEFAULTS = {
    "bbox_inches": "tight",
    "pad_inches": 0.02,
}


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
                | "SDF"
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


# TODO: merge together
def aligned_colorbar_old(
    ax,
    im,
    colorbar_label_size: int = FONT_SIZES["colorbar_label"],
    colorbar_tick_size: int = FONT_SIZES["colorbar_tick"],
    **kwargs,
):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.3, pad=0.05)
    cbar = ax.figure.colorbar(im, cax=cax, **kwargs)
    cbar.ax.tick_params(labelsize=colorbar_tick_size)
    if getattr(cbar, "ax", None) is not None and cbar.ax.get_ylabel():
        cbar.set_label(cbar.ax.get_ylabel(), fontsize=colorbar_label_size)
    return cbar


def aligned_colorbar(ax, im, **kwargs):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.3, pad=0.05)
    cb = ax.figure.colorbar(im, cax=cax, **kwargs)
    cb.ax.tick_params(labelsize=20)


# TODO: merge together
def plot_datapoint_old(
    name: str,
    datapoint: DataToVisualize,
    name_pic: str,
    settings_pic: dict,
    only_inner: bool = False,
    remove_axis: bool = False,
    dark_mode: bool = False,
):
    is_streamline = name.startswith("Streamline")

    if remove_axis and is_streamline:
        fig = Figure(figsize=(6, 5))
        # Use a copy to avoid side effects on the settings dict
        settings = settings_pic.copy()
        settings["dpi"] = 2560 / 5
        settings["bbox_inches"] = "tight"
        settings["pad_inches"] = 0

        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
    else:
        fig = Figure(figsize=(6.4, 5))
        settings = settings_pic
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(datapoint.category, fontsize=FONT_SIZES["title"])

    imshow_args = datapoint.imshowargs.copy()

    if imshow_args["cmap"] == "jp_temperature":
        if "error" in name.lower():
            _apply_temperature_cmap(datapoint, imshow_args)
        else:
            imshow_args["cmap"] = "RdBu_r"
    if dark_mode and is_streamline:
        imshow_args["cmap"] += "_dark"

    data_to_show = datapoint.data[100:400, 100:400] if only_inner else datapoint.data
    im = ax.imshow(data_to_show, **imshow_args)

    ax.invert_yaxis()

    if datapoint.vmax is not None and datapoint.vmin is not None:
        im.set_clim(datapoint.vmin, datapoint.vmax)

    if not (remove_axis and is_streamline):
        ax.set_ylabel("x [m]", fontsize=FONT_SIZES["axis_label"])
        ax.set_xlabel("y [m]", fontsize=FONT_SIZES["axis_label"])
        ax.tick_params(axis="both", which="major", labelsize=FONT_SIZES["tick_label"])
        aligned_colorbar(ax, im, label=datapoint.physical_property)
        fig.set_tight_layout(True)

    ext_inner = "_inner" if only_inner else ""
    save_settings = settings.copy()
    for key, value in SAVEFIG_DEFAULTS.items():
        save_settings.setdefault(key, value)
    fig.savefig(f"{name_pic}{ext_inner}.{settings['format']}", **save_settings)


def plot_datapoint(
    name: str,
    datapoint: DataToVisualize,
    name_pic: str,
    settings_pic: dict,
    only_inner: bool = False,
    remove_axis: bool = False,
    dark_mode: bool = False,
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


# TODO: merge together
def plot_datafields_old(
    data: dict[str, DataToVisualize],
    name_pic: str,
    settings_pic: dict,
    only_inner: bool = False,
    plot_all_in_1_pic: bool = True,
):
    if plot_all_in_1_pic:
        num_subplots = len(data)
        fig = Figure(figsize=(6.4, num_subplots * 3))

        axes = fig.subplots(num_subplots, 1, sharex=True)
        if num_subplots == 1:
            axes = [axes]

        for index, (_, datapoint) in enumerate(data.items()):
            ax = axes[index]
            ax.set_title(datapoint.category, fontsize=FONT_SIZES["title"])

            data_to_show = datapoint.data[100:400, 100:400] if only_inner else datapoint.data
            ax.imshow(data_to_show, **datapoint.imshowargs)

            ax.invert_yaxis()
            ax.set_ylabel("x [m]", fontsize=FONT_SIZES["axis_label"])
            ax.tick_params(axis="both", which="major", labelsize=FONT_SIZES["tick_label"])

        axes[-1].set_xlabel("y [m]", fontsize=FONT_SIZES["axis_label"])
        fig.set_tight_layout(True)

        ext_inner = "_inner" if only_inner else ""
        save_settings = settings_pic.copy()
        for key, value in SAVEFIG_DEFAULTS.items():
            save_settings.setdefault(key, value)
        fig.savefig(f"{name_pic}{ext_inner}.{settings_pic['format']}", **save_settings)
    else:
        for name, datapoint in data.items():
            plot_datapoint(name, datapoint, f"{name_pic}_{name}", settings_pic, only_inner)


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


def prepare_data_to_plot_sequential_outputs(
    y: torch.Tensor, y_out: torch.Tensor, plot_true: bool, info: dict, tiled: bool
):
    # prepare data of temperature true, temperature out, error, physical variables (inputs)
    # required_size = y_out.shape[-2:]
    # start_pos = ((y.shape[1] - required_size[1])//2, (y.shape[2] - required_size[2])//2)
    # y_reduced = y[:,start_pos[0]:start_pos[0]+required_size[1], start_pos[1]:start_pos[1]+required_size[2]]
    y_reduced = y.squeeze_(1)[..., 10:-10, 10:-10]
    y_out = y_out.squeeze_(1)[..., 10:-10, 10:-10]
    outs_max = [max(y_reduced.max(), y_out.max()) for idx in range(len(y_reduced))]
    outs_min = [min(y_reduced.min(), y_out.min()) for idx in range(len(y_reduced))]
    extent_highs_y = np.array(info["CellsSize"][:2]) * y_out.shape[-2:]

    dict_to_plot = {}
    labels = info["Labels"].keys()

    tiled_str = " tiled" if tiled else ""

    if y_reduced.dim() > 3:
        y_reduced = y_reduced.squeeze_(0)
    if y_out.dim() > 3:
        y_out = y_out.squeeze_(0)
    for label in labels:
        index = info["Labels"][label]["index"]
        for time_step in range(y_reduced.shape[0]):
            assert len(y_reduced[time_step].shape) == 2 and len(y_out[time_step].shape) == 2, (
                f"Shape are not 2D: {y_reduced[time_step].shape}, {y_out[time_step].shape}"
            )
            if plot_true:
                dict_to_plot[f"{label}_true_at_time_{time_step}{tiled_str}"] = DataToVisualize(
                    y_reduced[time_step], "Label", label, extent_highs_y, vmax=outs_max[index], vmin=outs_min[index]
                )
            dict_to_plot[f"{label}_out_at_time_{time_step}{tiled_str}"] = DataToVisualize(
                y_out[time_step], "Prediction", label, extent_highs_y, vmax=outs_max[index], vmin=outs_min[index]
            )
            dict_to_plot[f"{label}_error_at_time_{time_step}{tiled_str}"] = DataToVisualize(
                torch.abs(y_reduced[time_step] - y_out[time_step]), "Absolute Error", label, extent_highs_y
            )

    return dict_to_plot


def prepare_data_to_plot_sequential_inputs(x: torch.Tensor, y: torch.Tensor, info: dict):
    # prepare data of temperature true, temperature out, error, physical variables (inputs)
    # start_pos = ((y.shape[1] - required_size[1])//2, (y.shape[2] - required_size[2])//2)
    # y_reduced = y[:,start_pos[0]:start_pos[0]+required_size[1], start_pos[1]:start_pos[1]+required_size[2]]
    # y_reduced = y.squeeze_(1)

    dict_to_plot = {}

    inputs = info["Inputs"].keys()
    only_temp = False
    if not only_temp:
        for input_name in inputs:
            index = info["Inputs"][input_name]["index"]
            dict_to_plot[f"{input_name}"] = DataToVisualize(
                x[index, 0].squeeze_(), "Input", input_name, (np.array(info["CellsSize"][:2]) * x.shape[-2:])
            )

        for i, input_name in enumerate(["Absolute Times", "Gap times"]):
            for time in range(x.size(1)):
                dict_to_plot[f"{input_name}_t{time}"] = DataToVisualize(
                    x[i + len(inputs), time].squeeze_(),
                    f"Time step {time}",
                    input_name,
                    (np.array(info["CellsSize"][:2]) * x.shape[-2:]),
                )

    dict_to_plot["Temperature"] = DataToVisualize(
        x[-1].squeeze_(), "Input", "Temperature [C]", (np.array(info["CellsSize"][:2]) * x.shape[-2:])
    )

    return dict_to_plot


def plot_datafields_sequential(data: dict[str, DataToVisualize], name_pic: str, settings_pic: dict):
    """
    Groups entries by physical_property, then plots one figure per property
    with one subplot per timestep arranged horizontally.

    Expects keys in format "{input_name}_t{timestep}".
    """
    # Group by physical_property
    from collections import defaultdict

    groups = defaultdict(dict)  # property -> {timestep: DataToVisualize}

    for key, datapoint in data.items():
        # Extract timestep from key suffix "_t{n}"
        try:
            t = int(key.rsplit("_t", 1)[-1])
        except ValueError:
            t = 0
        groups[datapoint.physical_property][t] = datapoint

    for prop_name, timestep_dict in groups.items():
        timesteps = sorted(timestep_dict.keys())
        n_t = len(timesteps)

        fig = Figure(figsize=(4 * n_t, 4))
        axes = fig.subplots(1, n_t, sharey=True)
        if n_t == 1:
            axes = [axes]

        for ax, t in zip(axes, timesteps, strict=False):
            datapoint = timestep_dict[t]
            ax.set_title(f"{datapoint.category}", fontsize=FONT_SIZES["title"])  # e.g. "Time step 0"
            im = ax.imshow(datapoint.data, **datapoint.imshowargs)
            ax.invert_yaxis()
            ax.set_xlabel("y [m]", fontsize=FONT_SIZES["axis_label"])
            if t == timesteps[0]:
                ax.set_ylabel("x [m]", fontsize=FONT_SIZES["axis_label"])
            ax.tick_params(axis="both", which="major", labelsize=FONT_SIZES["tick_label"])
            aligned_colorbar(ax, im, label=prop_name)

        fig.suptitle(prop_name, fontsize=FONT_SIZES["suptitle"])
        fig.set_tight_layout(True)

        # Sanitize property name for use as filename
        safe_name = prop_name.replace("/", "_").replace(" ", "_").replace("[", "").replace("]", "")
        save_settings = settings_pic.copy()
        for key, value in SAVEFIG_DEFAULTS.items():
            save_settings.setdefault(key, value)
        fig.savefig(f"{name_pic}_{safe_name}.{settings_pic['format']}", **save_settings)


def reverse_norm_one_dp_sequence(x: torch.Tensor, y: torch.Tensor, y_out: torch.Tensor, norm: NormalizeTransform):
    # reverse transform for plotting real values
    x = norm.reverse(x.detach().cpu(), "Inputs")
    y = norm.reverse(y.detach().cpu(), "Labels")
    try:
        y_out = y_out.detach().cpu().squeeze(0)
    except (AttributeError, TypeError):
        y_out = y_out.squeeze(0)
    y_out = norm.reverse(y_out, "Labels")
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


def _region_has_values_over_threshold(
    sample: torch.Tensor,
    channel_idx: int = 5,
    x_range: tuple[int, int] = (20, 50),
    y_range: tuple[int, int] = (20, 50),
    threshold: float = 0.5,
) -> bool:
    if not isinstance(sample, torch.Tensor):
        return False

    data = sample
    if sample.dim() == 4:
        if sample.shape[1] > channel_idx:
            data = sample[0, channel_idx]
        elif sample.shape[0] > channel_idx:
            data = sample[channel_idx]
        else:
            return False
    elif sample.dim() == 3:
        if sample.shape[0] <= channel_idx:
            return False
        data = sample[channel_idx]
    elif sample.dim() != 2:
        return False

    x0, x1 = x_range
    y0, y1 = y_range
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(data.shape[0] - 1, x1)
    y1 = min(data.shape[1] - 1, y1)
    if x1 < x0 or y1 < y0:
        return False

    region = data[x0 : x1 + 1, y0 : y1 + 1]
    return region.max().item() > threshold


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


def plot_output_over_input(
    input_data: dict[str, DataToVisualize], output_data: dict[str, DataToVisualize], name_pic: str, settings_pic: dict
):
    """Plot output data overlaid on top of input data with transparency."""
    num_subplots = len(input_data)
    fig = Figure(figsize=(6.4, num_subplots * 3))

    axes = fig.subplots(num_subplots, 1, sharex=True)
    if num_subplots == 1:
        axes = [axes]

    for index, (name, input_datapoint) in enumerate(input_data.items()):
        imshow_args = input_datapoint.imshowargs.copy()

        if imshow_args["cmap"] == "jp_temperature":
            # TODO: in einer neueren Version habe sind datasetTypes eingeführt worden
            _apply_temperature_cmap(input_datapoint, imshow_args)

        ax = axes[index]
        ax.set_title(
            f"{input_datapoint.category} (Input: {input_datapoint.physical_property})", fontsize=FONT_SIZES["title"]
        )

        # Resolve output datapoint robustly: direct-key match, same property, or fallback to temperature.
        output_datapoint = None
        candidate_names = [name, name.replace("_true", "_out"), name.replace("_out", "_true")]
        for candidate in candidate_names:
            if candidate in output_data:
                output_datapoint = output_data[candidate]
                break

        if output_datapoint is None:
            for candidate in output_data.values():
                if candidate.physical_property == input_datapoint.physical_property:
                    output_datapoint = candidate
                    break

        if output_datapoint is None:
            for candidate in output_data.values():
                if candidate.physical_property == "Temperature [C]":
                    output_datapoint = candidate
                    break

        if output_datapoint is not None:
            imshow_args = output_datapoint.imshowargs.copy()

            if imshow_args["cmap"] == "jp_temperature":
                _apply_temperature_cmap(output_datapoint, imshow_args)

        ax.invert_yaxis()
        ax.set_ylabel("x [m]", fontsize=FONT_SIZES["axis_label"])
        ax.tick_params(axis="both", which="major", labelsize=FONT_SIZES["tick_label"])

    axes[-1].set_xlabel("y [m]", fontsize=FONT_SIZES["axis_label"])
    fig.set_tight_layout(True)
    save_settings = settings_pic.copy()
    for key, value in SAVEFIG_DEFAULTS.items():
        save_settings.setdefault(key, value)
    fig.savefig(f"{name_pic}.{settings_pic['format']}", **save_settings)


# TODO: merge together
def visualize_inputs_old(
    datasetType: DatasetType,
    dataloader,
    args: dict,
    amount_datapoints_to_visu: int = inf,
    plot_path: str = "default",
    pic_format: str = "png",
    target_match: int = 1,
):
    log.info("Visualizing Inputs...")

    try:
        norm = dataloader.dataset.norm
        info = dataloader.dataset.info
        dataset = dataloader.dataset
    except AttributeError:
        norm = dataloader.dataset.dataset.norm
        info = dataloader.dataset.dataset.info
        dataset = dataloader.dataset.dataset

    settings_pic = {"format": pic_format, "dpi": 160}
    is_sequential = dataset.__class__.__name__ in ["SimulationDatasetCutsSequential", "DataPointSequence", "Subset"]
    n_subsets = len(dataset.subsets) if (is_sequential and hasattr(dataset, "subsets")) else 1

    # Accumulate samples grouped by chain_idx
    # chain_buffer: chain_idx -> {subset_idx: (x, y)}
    chain_buffer = {}
    plotted_count = 0

    for inputs, labels, metadata_list in dataloader:
        batch_size = inputs.shape[0]

        for i in range(batch_size):
            if is_sequential and n_subsets > 1:
                chain_idx = metadata_list["chain_idx"][i].item()
                subset_idx = metadata_list["subset_idx"][i].item()
            else:
                chain_idx = plotted_count  # treat each sample as its own chain
                subset_idx = 0

            if chain_idx not in chain_buffer:
                chain_buffer[chain_idx] = {}
            chain_buffer[chain_idx][subset_idx] = (inputs[i], labels[i])

            # Only plot when we have collected all subsets for this chain
            if len(chain_buffer[chain_idx]) < n_subsets:
                continue

            if plotted_count >= amount_datapoints_to_visu:
                return

            # --- Build combined dict across all subsets ---
            combined_dict = {}
            for s_idx in sorted(chain_buffer[chain_idx].keys()):
                x_s, y_s = chain_buffer[chain_idx][s_idx]
                x_rev, y_rev = reverse_norm_one_dp_inputs(x_s, y_s, norm)

                if is_sequential:
                    sub_dict = prepare_data_to_plot_sequential_inputs(x_rev, y_rev, info)
                    # Prefix keys with subset index so plots don't overwrite each other
                    sub_dict = {f"subset{s_idx}_{k}": v for k, v in sub_dict.items()}
                else:
                    sub_dict = prepare_data_to_plot_inputs(x_rev, y_rev, info)

                combined_dict.update(sub_dict)

            name_pic = f"{plot_path}_{plotted_count}_input"
            plot_datafields(combined_dict, name_pic, settings_pic, only_inner=False, plot_all_in_1_pic=False)

            del chain_buffer[chain_idx]  # free memory
            plotted_count += 1


def visualize_inputs(
    datasetType: DatasetType,
    dataloader,
    args: dict,
    amount_datapoints_to_visu: int = inf,
    plot_path: str = "default",
    pic_format: str = "png",
):
    log.info("Visualizing Inputs...")

    total_samples = len(dataloader.dataset)
    limit = min(amount_datapoints_to_visu, total_samples)

    try:
        norm = dataloader.dataset.norm
        info = dataloader.dataset.info
        dataset = dataloader.dataset
    except AttributeError:
        norm = dataloader.dataset.dataset.norm
        info = dataloader.dataset.dataset.info
        dataset = dataloader.dataset.dataset

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


# TODO: merge together
def visualize_outputs_old(
    datasetType: DatasetType,
    model,
    dataloader,
    args: dict,
    amount_datapoints_to_visu: int = inf,
    plot_path: str = "default",
    pic_format: str = "png",
    scaleBounds=None,
    useNonLinearCmap: bool = None,
    target_match: int = 1,
    plot_true: bool = True,
):
    log.info("Visualizing Outputs...")

    try:
        norm = dataloader.dataset.norm
        info = dataloader.dataset.info
        dataset = dataloader.dataset
    except AttributeError:
        norm = dataloader.dataset.dataset.norm
        info = dataloader.dataset.dataset.info
        dataset = dataloader.dataset.dataset

    settings_pic = {"format": pic_format, "dpi": 160}
    device = args["device"]
    is_sequential = dataset.__class__.__name__ in ["SimulationDatasetCutsSequential", "DataPointSequence", "Subset"]
    n_subsets = len(dataset.subsets) if (is_sequential and hasattr(dataset, "subsets")) else 1

    chain_buffer = {}  # chain_idx -> {subset_idx: (x, y)}
    plotted_count = 0

    for inputs, labels, metadata_list in dataloader:
        batch_size = inputs.shape[0]

        for i in range(batch_size):
            if is_sequential and n_subsets > 1:
                chain_idx = metadata_list["chain_idx"][i].item()
                subset_idx = metadata_list["subset_idx"][i].item()
            else:
                chain_idx = plotted_count
                subset_idx = 0

            if chain_idx not in chain_buffer:
                chain_buffer[chain_idx] = {}
            chain_buffer[chain_idx][subset_idx] = (inputs[i], labels[i])

            if len(chain_buffer[chain_idx]) < n_subsets:
                continue

            if plotted_count >= amount_datapoints_to_visu:
                return

            # --- Run inference and build combined dict across all subsets ---
            combined_dict = {}
            init_frame = None
            for s_idx in sorted(chain_buffer[chain_idx].keys()):
                x_s, y_s = chain_buffer[chain_idx][s_idx]

                tiled = x_s.shape[-1] > 1000 or x_s.shape[-2] > 1000
                if is_sequential:
                    if tiled:
                        y_out_raw = model.infer_tiled(x_s.unsqueeze(0), device)
                    else:
                        y_out_raw = model.infer(x_s.unsqueeze(0), device, init_frame=init_frame)

                    init_frame = y_out_raw[:, :, -1]
                    x_rev, y_rev, y_out_rev = reverse_norm_one_dp_sequence(x_s, y_s, y_out_raw, norm)
                    sub_dict = prepare_data_to_plot_sequential_outputs(y_rev, y_out_rev, plot_true, info, tiled)
                else:
                    y_out_raw = model.infer(x_s.unsqueeze(0), device)
                    y_out_rev = reverse_norm_one_dp_outputs(y_out_raw, norm)
                    _, y_rev = reverse_norm_one_dp_inputs(x_s, y_s, norm)
                    sub_dict = prepare_data_to_plot_outputs(y_rev, y_out_rev, info)

                # Prefix to keep subsets distinct in the combined plot
                sub_dict = {f"subset{s_idx}_{k}": v for k, v in sub_dict.items()}
                combined_dict.update(sub_dict)

            name_pic = f"{plot_path}_{plotted_count}_output"
            plot_datafields(combined_dict, name_pic, settings_pic, only_inner=False, plot_all_in_1_pic=False)

            del chain_buffer[chain_idx]
            plotted_count += 1


def visualize_outputs(
    datasetType: DatasetType,
    model,
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
        dataset = dataloader.dataset
    except AttributeError:
        norm = dataloader.dataset.dataset.norm
        info = dataloader.dataset.dataset.info
        dataset = dataloader.dataset.dataset

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


def visualize_outputs_over_inputs(
    model: UNet,
    dataloader,
    args: dict,
    amount_datapoints_to_visu: int = inf,
    plot_path: str = "default",
    pic_format: str = "png",
):
    log.info("Visualizing Outputs over Inputs...")
    total_samples = len(dataloader.dataset)
    limit = min(amount_datapoints_to_visu, total_samples)
    log.info(f"Total samples: {total_samples}, Limit: {limit}")

    try:
        norm = dataloader.dataset.norm
        info = dataloader.dataset.info
    except AttributeError:
        norm = dataloader.dataset.dataset.norm
        info = dataloader.dataset.dataset.info

    log.info("Normalization and info loaded successfully")

    settings_pic = {"format": pic_format, "dpi": 160}
    current_count = 0
    device = args["device"]
    log.info(f"Device: {device}, plot_path: {plot_path}")

    overfit = args.get("overfit", False)
    overfit_on = args.get("overfit_on", None)
    log.info(f"Overfit mode: {overfit}, overfit_on: {overfit_on}")

    for batch, (inputs, labels, _) in enumerate(dataloader):
        batch_size = inputs.shape[0]
        log.info(f"Processing batch {batch}, batch_size: {batch_size}")

        for i in range(batch_size):
            if current_count >= limit:
                log.info(f"Reached limit ({limit}), stopping visualization")
                return

            threshold_check = _label_has_midcell_above_threshold(labels[i], 0.5)
            is_val = "val" in str(plot_path)
            log.info(
                f"Sample {current_count} (batch {batch}, idx {i}): threshold_check={threshold_check}, is_val={is_val}"
            )

            if not threshold_check and not is_val:
                log.info("  -> Skipped (threshold not met and not validation set)")
                continue

            log.info(f"  -> Processing sample {current_count}")

            x = inputs[i]
            y = labels[i]

            dataset_class = dataloader.dataset.__class__.__name__
            log.info(f"  Dataset class: {dataset_class}")

            if dataset_class in ["SimulationDatasetCutsSequential", "DataPointSequence", "Subset"]:
                log.info("  -> Using sequential data path")
                # Generate prediction
                y_out = model.infer(x.unsqueeze(0), device)

                # Reverse normalization
                x_t_denorm, y_denorm = reverse_norm_one_dp_inputs(x, y, norm)
                y_out_denorm = reverse_norm_one_dp_outputs(y_out, norm)

                # Prepare input data for plotting
                input_dict = prepare_data_to_plot_sequential_inputs(x_t_denorm, y_denorm, info)
                log.info(f"  Input dict keys: {list(input_dict.keys())}")

                # Prepare output data for plotting
                output_dict = prepare_data_to_plot_sequential_outputs(y_denorm, y_out_denorm, info)
                log.info(f"  Output dict keys (before filtering): {list(output_dict.keys())}")

                temp_true_dict = {
                    k: v
                    for k, v in output_dict.items()
                    if v.physical_property == "Temperature [C]" and v.category == "Label"
                }
                log.info(f"  Temp true dict keys: {list(temp_true_dict.keys())}")

                if temp_true_dict:
                    output_dict = temp_true_dict
                    log.info("  Using temp_true_dict")
                else:
                    output_dict = {k: v for k, v in output_dict.items() if v.physical_property == "Temperature [C]"}
                    log.info(f"  Using filtered output_dict with Temperature [C]: {list(output_dict.keys())}")

                # Plot output overlaid on input
                name_pic = f"{plot_path}_{current_count}_output_over_input"
                log.info(f"  Saving plot to: {name_pic}")
                plot_output_over_input(input_dict, output_dict, name_pic, settings_pic)
                log.info("  Plot saved successfully")
            else:
                log.info("  -> Using non-sequential data path")
                # For non-sequential data, just use the single input
                x_copy = deepcopy(x)
                y_copy = deepcopy(y)

                y_out_raw = model.infer(x_copy.unsqueeze(0), device)

                y_out = reverse_norm_one_dp_outputs(y_out_raw, norm)
                x_denorm, y_denorm = reverse_norm_one_dp_inputs(x_copy, y_copy, norm)

                # Prepare input and output data for plotting
                input_dict = prepare_data_to_plot_inputs(x_denorm, y_denorm, info)
                output_dict = prepare_data_to_plot_outputs(y_denorm, y_out, info, lic=False)
                temp_true_dict = {
                    k: v
                    for k, v in output_dict.items()
                    if v.physical_property == "Temperature [C]" and v.category == "Label"
                }
                if temp_true_dict:
                    output_dict = temp_true_dict
                else:
                    output_dict = {k: v for k, v in output_dict.items() if v.physical_property == "Temperature [C]"}

                name_pic = f"{plot_path}_{current_count}_output_over_input"
                plot_output_over_input(input_dict, output_dict, name_pic, settings_pic)

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
