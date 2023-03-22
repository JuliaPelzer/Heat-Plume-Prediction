import data.dataset_loading as lp
from utils.visualize_data import View, _aligned_colorbar
from dataclasses import dataclass
import sys
import logging
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
from data.utils import separate_property_unit


def what_do_we_have(dataset_name="groundtruth_hps_no_hps/test_bc", path_to_datasets="/home/pelzerja/Development/simulation_groundtruth_pflotran/Phd_simulation_groundtruth"):
    """
    Initialize dataset and dataloader for training or just plotting.

    Parameters
    ----------
        dataset_name : str
            Name of the dataset to use (has to be the same as in the folder)
        path_to_datasets : str
            Path to the datasets, together with dataset_name full path    

    Returns
    -------
        datasets : dict
            Dictionary of datasets, with keys "train", "val", "test"
    """
    
    logging.basicConfig(level=logging.WARNING)        # level: DEBUG, INFO, WARNING, ERROR, CRITICAL

    # load whole dataset, splits already in different modes
    if path_to_datasets:
        if dataset_name:
            datasets, _ = lp.init_data(reduce_to_2D=False, just_plotting=True, dataset_name=dataset_name, path_to_datasets=path_to_datasets)
        else:
            datasets, _ = lp.init_data(reduce_to_2D=False, just_plotting=True, path_to_datasets=path_to_datasets)
    else:
        if dataset_name:
            datasets, _ = lp.init_data(reduce_to_2D=False, just_plotting=True, dataset_name=dataset_name)
        else:
            datasets, _ = lp.init_data(reduce_to_2D=False, just_plotting=True)



    # run this to see which RUN_x is loaded into which data point (datasets["train"][x])
    i = 0
    for run in datasets["train"]:
        logging.warning(f"datapoint in train {i}: {run['run_id']}")
        i += 1

    return datasets


#----------------------------------------------------------------------------------------------------------------------

@dataclass
class View:
    name:str
    x_label:str
    y_label:str
    cut_slice:np.array=None
    transpose:bool=False

def plot_datapoint(datasets, data_point_number=0, view="top"):
    """
    Plot a single data point from the dataset.
    Parameters
    ----------
    datasets : dict
        Dictionary of datasets, with keys "train", "val", "test", 
        Format: datasets["train"][data_point_number] = {'x': tensor, 'y': tensor, 'run_id': int}
        with "x" / "y": torch.Tensor of shape (channels, height, width, depth)
    data_point_number : int
        Number of the data point to plot
    view : str
        View (cut through the 3D groundwater body) of the data point to visualize, 
        choose from {top, topish, top_hp, side, side_hp}
    """

    # decide which data point to visualize
    tensor_dict = datasets["train"][data_point_number]

    # resulting pic is in "visualization/pics/plot_phys_props_RUN_{run_id}_VIEW_{view}.jpg"
    prop_in = datasets["train"].get_input_properties()
    prop_out = datasets["train"].get_output_properties()
    plot_data_inner(tensor_dict, prop_in, prop_out, view=view, oriented="left")


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

        title =     title =  prefix + separate_property_unit(property_names[channel])[0]
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

# data structure of datasets:
# 

# FYI: STILL IN PROCESS, CHANGES WILL FOLLOW, DONT HOLD ME ACCOUNTABLE FOR THIS


if __name__ == "__main__":
    #path="/home/pelzerja/Development/simulation_groundtruth_pflotran/Phd_simulation_groundtruth"
    #"groundtruth_hps_no_hps/test_bc"
    print("In which folder are your datasets?")
    path_to_datasets = input()
    print("Which dataset do you want to visualize?")
    dataset_name = input()

    datasets = what_do_we_have(dataset_name=dataset_name, path_to_datasets=path_to_datasets)

    print("Which data point do you want to visualize? Please name the number directly behind 'train'")
    data_point_number = input()
    print("Which view do you want to visualize? Please choose from {top, topish, top_hp, side, side_hp}.")
    view = input()

    plot_datapoint(datasets, data_point_number=int(data_point_number), view=view)

    print("Done")
