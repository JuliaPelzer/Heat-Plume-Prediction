from dataclasses import dataclass
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import List
import torch
from data.dataset import DatasetSimulationData
from data.utils import separate_property_unit
# from torch.utils.tensorboard import SummaryWriter
from data.dataloader import DataLoader
from networks.unet_leiterrl import UNet
from data.utils import PhysicalVariables
from torch.nn import MSELoss

# TODO: look at vispy library for plotting 3D data

@dataclass
class View:
    name:str
    x_label:str
    y_label:str
    cut_slice:np.array=None
    transpose:bool=False

def plot_datapoint(dataset : DatasetSimulationData, run_id : int, view="top", plot_streamlines=False, oriented="center") -> None:
    """
    Plot all physical properties of one data point, depending on the `view` with streamlines

    Parameters
    ----------
        dataset : DatasetSimulationData
            Dataset to take the data point and information about the physical properties from
        run_id : int
            Index of the data point to plot
        view : str
            From which view to plot (top vs. side, outside vs. height of heat pipe)

    Returns
    -------
        None
    """

    plot_data_inner(data=dataset[run_id], property_names_in=dataset.get_input_properties(), property_names_out=dataset.get_output_properties(), run_id=run_id, view=view, plot_streamlines=plot_streamlines, oriented=oriented)


def plot_data_inner(data : Dict[str, np.ndarray], property_names_in : List[str], property_names_out : List[str], run_id:int=42, view:str="top",
    plot_streamlines=False, oriented="center"):
    # function excluded to also be able to plot the reversed dataset, #TODO make reversing cleaner so this step is unnecessary
    assert view in ["top", "topish", "top_hp", "side", "side_hp"], "view must be one of 'top', 'topish', 'top_hp', 'side', 'side_hp'"
    assert isinstance(run_id, int), "run_id must be an integer"

    index_overall = 0
    n_dims = len(property_names_in) + len(property_names_out)
    n_subplots = n_dims + 1 if plot_streamlines else n_dims

    _, axes = plt.subplots(n_subplots,1,sharex=True,figsize=(20,3*(n_subplots)))

    if oriented=="center":
        cut_x_hp = 7
    elif oriented=="left":
        cut_x_hp = 9
    # dictionary of all potential views with respective labels and positions where to slice the data
    view_dict = {"top": View(name="top", x_label="y", y_label="x", cut_slice=np.s_[:, :, -1], transpose=False),
                "top_hp": View(name="top_hp", x_label="y", y_label="x", cut_slice=np.s_[:, :, 9], transpose=False),
                "topish": View(name="topish", x_label="y", y_label="x", cut_slice=np.s_[:, :, -3], transpose=False),
                "side": View(name="side", x_label="y", y_label="z", cut_slice=np.s_[11, :, :], transpose=True),
                "side_hp": View(name="side_hp", x_label="y", y_label="z", cut_slice=np.s_[cut_x_hp, :, :], transpose=True)}

    index_overall = _plot_properties(data['x'], index_overall, view_dict[view], axes, property_names=property_names_in, prefix = "Input ")
    index_overall = _plot_properties(data['y'], index_overall, view_dict[view], axes, property_names=property_names_out, prefix = "Output ")
    if plot_streamlines:
        index_overall = _plot_streamlines(data['y'], index_overall, view_dict[view], axes, property_names=property_names_in)

    pic_file_name = "visualization/pics/plot_phys_props"
    if plot_streamlines:
        pic_file_name += "_with_streamlines"
    if run_id:
        pic_file_name += "_RUN_" + str(run_id)
    pic_file_name += "_VIEW_" + view + ".png"
    print(f"Resulting picture is at {pic_file_name}")
    plt.savefig(pic_file_name)

def plot_sample(model:UNet, dataloader: DataLoader, device:str, name_folder:str, amount_plots:int=None, plot_name:str="plot_learned_test_sample"):

    # writer = SummaryWriter(f"runs/{name_folder}")
    error = []
    error_mean = []
    reverse_done = False

    if amount_plots is None:
        amount_plots = len(dataloader.dataset)

    for batch_id, (inputs,labels) in enumerate(dataloader):
        len_batch = inputs.shape[0]
        for datapoint_id in range(len_batch):
            x = inputs[datapoint_id].to(device)
            x = torch.unsqueeze(x,0)
            y = labels[datapoint_id].to(device)
            y = torch.unsqueeze(y,0)
            y_out = model(x).to(device)

            # reverse transform for plotting real values
            x = dataloader.reverse_transform(x)
            y = dataloader.reverse_transform_temperature(y)
            y_out = dataloader.reverse_transform_temperature(y_out)

            try:
                physical_vars = dataloader.dataset.datapoints[datapoint_id].inputs.keys()
            except:
                physical_vars = dataloader.dataset[datapoint_id].inputs.keys()

            error_current = y-y_out
            temp_max = max(y.max(), y_out.max())
            temp_min = min(y.min(), y_out.min())
            list_to_plot = [
                _make_dict_batchbased(y.detach().cpu(), "temperature true", 0),# vmax=temp_max, vmin=temp_min),
                _make_dict_batchbased(y_out.detach().cpu(), "temperature out", 0),# vmax=temp_max, vmin=temp_min),
                _make_dict_batchbased(error_current.detach().cpu(), "error", 0),
            ]

            for index, physical_var in enumerate(physical_vars):
                list_to_plot.append(_make_dict_batchbased(x.detach().cpu(), physical_var, index))

            current_id = datapoint_id + batch_id*len_batch
            _plot_y(list_to_plot, title=name_folder, name_pic=plot_name+"_"+str(current_id))

            error.append(abs(error_current.detach()))
            error_mean.append(np.mean(error_current.cpu().numpy()).item())

            if current_id == amount_plots-1:
                break

    max_error = np.max(error[-1].cpu().numpy())
    print("Maximum error: ", max_error)
        # writer.close()
    return error, error_mean, max_error

def plot_exemplary_learned_result_OLD(model, dataloaders, name_pic="plot_y_exemplary"):
    """not pretty but functional to get a first glimpse of how y_out looks compared to y_truth"""

    for data in dataloaders["train"]:
        x_exemplary = data["x"].float()
        y_true_exemplary = data["y"].float()
        y_out_exemplary = model(x_exemplary)
        # writer.add_image("y_out_test_0", y_out_test[0,0,:,:], dataformats="WH", global_step=0)
        # writer.add_image("y_true_test_0", y_true_test[0,0,:,:], dataformats="WH", global_step=0)
        # mse_loss = loss_fn(y_out_test, y_true_test)
        # loss = mse_loss
        # writer.add_scalar("loss", loss.item(), 0)
        break

    # list_to_plot = [
    #     make_dict(y_true_exemplary, "temperature", 0),
    #     make_dict(y_out_exemplary, "temperature", 0),
    #     make_dict(x_exemplary, "y_velocity", 0),
    #     make_dict(x_exemplary, "z_velocity", 1),
    #     make_dict(x_exemplary, "pressure", 2),
    #     make_dict(x_exemplary, "hp location", 3),
    #     make_dict(x_exemplary, "init temperature", 4),
    # ]
    error = y_true_exemplary-y_out_exemplary
    # error_abs = torch.abs(error)
    list_to_plot = [
        _make_dict_batchbased(y_true_exemplary, "temperature true", 0),
        _make_dict_batchbased(y_out_exemplary, "temperature out", 0),
        _make_dict_batchbased(error, "error", 0),
        #_make_dict(error_abs, "error abs", 0),
        _make_dict_batchbased(x_exemplary, "pressure", 0),
        _make_dict_batchbased(x_exemplary, "hp location", 1),
    ]

    _plot_y(list_to_plot, title=name_pic, name_pic=name_pic)

    return error

## helper functions for plotting
def _plot_properties(data : np.ndarray, index_overall:int, view: View, axes, property_names : List[str], prefix:str = "") -> int:
    """
    Plot all properties of one data point

    Parameters
    ----------
        data : np.ndarray
            Datapoint to plot, dimensions: channels x HxWxD
        property_names : List[str]
            List of all properties to plot
        prefix : str
            Prefix to add to the title of the plot like "Input " or "Output "

    Returns
    -------
        None

    """

    for channel in np.arange(len(data)):
        plt.sca(axes[index_overall])
        field = data[channel, :, :, :]
        if len(field.shape) != 3:
            raise ValueError("Data is not 3D")

        title = _build_title(prefix, property_names, channel)
        plt.title(title)
        # plot field, if views transpose is true, transpose the field
        if view.transpose:
            plt.imshow(field[view.cut_slice].T)
            plt.gca().invert_yaxis()

        else:
            plt.imshow(field[view.cut_slice])

        plt.xlabel(view.x_label)
        plt.ylabel(view.y_label)

        _aligned_colorbar(label=property_names[channel])
        index_overall += 1

    return index_overall

def _plot_streamlines(data : np.ndarray, index_overall:int, view: View, axes, property_names: List[str]) -> int:
    # TODO fix input, output vars format PhysicalVariables
    assert "Liquid Y-Velocity [m_per_y]" in property_names and "Liquid Z-Velocity [m_per_y]" in property_names, "Y- and Z-Velocity are not in the properties"
    index_x_velocity = property_names.index("Liquid X-Velocity [m_per_y]")
    index_y_velocity = property_names.index("Liquid Y-Velocity [m_per_y]")
    index_z_velocity = property_names.index("Liquid Z-Velocity [m_per_y]")

    plt.sca(axes[index_overall])
    if view.name[0:4]=="side":
        field_U = data[index_y_velocity, :, :, :] # "Liquid Y-Velocity [m_per_y]"
        field_V = data[index_z_velocity, :, :, :] # "Liquid Z-Velocity [m_per_y]"
        Z, Y = np.mgrid[0:len(field_U[0,0,:]),0:len(field_U[0,:,0])]
        plt.title("Streamlines of Y,Z-Velocity")

    elif view.name[0:3]=="top":
        field_U = data[index_y_velocity, :, :, :] # "Liquid Y-Velocity [m_per_y]"
        field_V = data[index_x_velocity, :, :, :] # "Liquid Z-Velocity [m_per_y]"
        Z, Y = np.mgrid[0:len(field_U[:,0,0]),0:len(field_U[0,:,0])]
        plt.title("Streamlines of X,Y-Velocity")

    assert field_U.shape == field_V.shape, "Velocity-fields have different shapes"
    assert field_U.shape != 3, "Data is not 3D"

    if view.transpose:
        U = field_U[view.cut_slice].T
        V = field_V[view.cut_slice].T
    else:
        U = field_U[view.cut_slice]
        V = field_V[view.cut_slice]

    plt.streamplot(Y, Z, U, V, density=[1.2, 1.2], arrowstyle="->", broken_streamlines=False)
    if not view.transpose:
        plt.gca().invert_yaxis()
    plt.xlabel(view.x_label)
    plt.ylabel(view.y_label)
    _aligned_colorbar(label="Y,Z-Velocity [m_per_year]")

    index_overall += 1

    return index_overall

def _build_title(prefix:str, property_names:List[str], channel:int) -> str:
    """
    Build title for plot by removing the unit

    Parameters
    ----------
        prefix : str
            Prefix to add to the title of the plot like "Input " or "Output "
        property_names : List[str]
            List of all properties to plot
        channel : int
            Index of current property to plot

    Returns
    -------
        title : str

    """

    title =  prefix + separate_property_unit(property_names[channel])[0]
    print(title)

    return title

def _make_dict_batchbased(data:np.ndarray, physical_property:str, index:int, **imshowargs):
    data_dict = {"data" : data, "property" : physical_property, "imshowargs" : imshowargs}
    data_dict["data"] = data_dict["data"][0,index,:,:]
    return data_dict

def _make_dict_datapointbased(data:PhysicalVariables, physical_property:str, **imshowargs):
    data_dict = {"data" : data[physical_property].value, "property" : physical_property, "imshowargs" : imshowargs}
    return data_dict

def _plot_y(data, title, name_pic="plot_y_exemplary"):
    n_subplots = len(data)
    _, axes = plt.subplots(n_subplots,1,sharex=True,figsize=(20,3*(n_subplots)))
    plt.title(title)

    for index, data_point in enumerate(data):
        plt.sca(axes[index])
        plt.imshow(data_point["data"].T, **data_point["imshowargs"])
        plt.gca().invert_yaxis()

        plt.xlabel("y")
        plt.ylabel("z")
        _aligned_colorbar(label=data_point["property"])

    pic_file_name = f"runs/{name_pic}.png"
    print(f"Resulting picture is at {pic_file_name}")
    plt.savefig(pic_file_name)

def _aligned_colorbar(*args,**kwargs):
    cax = make_axes_locatable(plt.gca()).append_axes("right",size= 0.3,pad= 0.05)
    plt.colorbar(*args,cax=cax,**kwargs)

## helper to find inlet, outlet alias max and min Material_ID
# print(np.where(data["Material_ID"]==np.max(data["Material_ID"])))

# def slice_y(y, property):
#     property = 0 if property == "temperature" else property #1
#     return y.detach().numpy()[0,property,:,:]