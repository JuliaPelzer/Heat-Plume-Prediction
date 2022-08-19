import os
from typing import Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py
from tqdm.auto import tqdm
from typing import List

from data.dataset import GWF_HP_Dataset
import torch

# TODO: look at vispy library for plotting 3D data

## helper function for plotting
def aligned_colorbar(*args,**kwargs):
    cax = make_axes_locatable(plt.gca()).append_axes("right",size= 0.3,pad= 0.05)
    plt.colorbar(*args,cax=cax,**kwargs)
    
def plot_datapoint(dataset : GWF_HP_Dataset, run_id : int, view="top", prefix="", plot_streamlines=False, oriented="center") -> None:
    """
    Plot all physical properties of one data point, depending on the `view` also with streamlines.. if they work at some time...
    
    Parameters
    ----------
        dataset : GWF_HP_Dataset
            Dataset to take the data point and information about the physical properties from
        run_id : int
            Index of the data point to plot
        view : str
            From which view to plot from (top vs. side, outside vs. height of heat pipe)

    Returns
    -------
        None
    """
    property_names_in = dataset.get_input_properties()
    property_names_out = dataset.get_output_properties()
    data = dataset[run_id]

    plot_data_inner(data=data, property_names_in=property_names_in, property_names_out=property_names_out, run_id=run_id, view=view, plot_streamlines=plot_streamlines, oriented=oriented)

def plot_data_inner(data : Dict[str, np.ndarray], property_names_in : List[str], property_names_out : List[str], run_id=42, view="top", plot_streamlines=False, oriented="center") -> None:
    # function excluded to also be able to plot the reversed dataset, #TODO make reversing cleaner so this step is unnecessary
    index_overall = 0
    n_dims = len(property_names_in) + len(property_names_out)
    n_subplots = n_dims + 1 if plot_streamlines else n_dims
    
    fig, axes = plt.subplots(n_subplots,1,sharex=True,figsize=(20,3*(n_subplots)))

    if oriented=="center":
        cut_x_hp = 7
    elif oriented=="left":
        cut_x_hp = 9
    # dictionary of all potential views with respective labels and positions where to slice the data 
    view_dict = {
        "top": {
            "x_label": "y",
            "y_label": "x",
            "cut_slice": np.s_[:,:,-1],
            "transpose": False
        },
        "top_hp": {
            "x_label": "y",
            "y_label": "x",
            "cut_slice": np.s_[:,:,9], #8
            "transpose" : False
        },
        "topish": {
            "x_label": "y",
            "y_label": "x",
            "cut_slice": np.s_[:,:,-3],
            "transpose" : False
        },
        "side": {
            "x_label": "y",
            "y_label": "z",
            "cut_slice": np.s_[11,:,:], # Tensor
            # Numpy: "cut_slice": np.s_[11,:,::-1]
            "transpose" : True
        },
        "side_hp": {
            "x_label": "y",
            "y_label": "z",
            "cut_slice": np.s_[cut_x_hp,:,:], # Tensor
            # Numpy: "cut_slice": np.s_[9,:,::-1],
            "transpose" : True
        }
    }

    def plot_properties(data : np.ndarray, property_names : List[str], prefix = "") -> None:
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
        nonlocal index_overall

        for channel in np.arange(len(data)):
            plt.sca(axes[index_overall])
            field = data[channel, :, :, :]
            if len(field.shape) != 3:
                raise ValueError("Data is not 3D")

            index = property_names[channel].find(' [')
            title = prefix + property_names[channel]
            if index != -1:
                title = prefix + property_names[channel][:index]
            plt.title(title)
            # plot field, if view_dict transpose is true, transpose the field
            if view_dict[view]["transpose"]:
                plt.imshow(field[view_dict[view]["cut_slice"]].T)
                plt.gca().invert_yaxis()

            else:
                plt.imshow(field[view_dict[view]["cut_slice"]])
            
            plt.xlabel(view_dict[view]["x_label"])
            plt.ylabel(view_dict[view]["y_label"])

            aligned_colorbar(label=property_names[channel])
            index_overall += 1
            #print(torch.mean(field), torch.std(field))
    
    plot_properties(data['x'], property_names=property_names_in, prefix = "Input ")
    plot_properties(data['y'], property_names=property_names_out, prefix = "Output ")

    #streamlines
    if plot_streamlines:
        plt.sca(axes[index_overall])
        plt.title("Streamlines")
        #TODO FIELD? REINPACKEN IN die DEF?
        #if view=="side":
        #    Z, Y = np.mgrid[0:len(field[0,0,:]),0:len(field[0,:,0])]
        #    U = df.at[run_id,'Liquid Y-Velocity [m_per_y]'][11,:,::-1]
        #    V = df.at[run_id,'Liquid Z-Velocity [m_per_y]'][11,:,::-1]
        #    plt.streamplot(Y, Z, U.T, V.T, density=[2, 0.7])
        #    plt.xlabel("y")
        #    plt.ylabel("z")
        #elif view=="side_hp":
        #    Z, Y = np.mgrid[0:len(field[0,0,:]),0:len(field[0,:,0])]
        #    U = df.at[run_id,'Liquid Y-Velocity [m_per_y]'][8,:,::-1]
        #    V = df.at[run_id,'Liquid Z-Velocity [m_per_y]'][8,:,::-1]
        #    plt.streamplot(Y, Z, U.T, V.T, density=[2, 0.7])
        #    plt.xlabel("y")
        #    plt.ylabel("z")
        #elif view=="top_hp":
        #    X, Y = np.mgrid[0:len(field[:,0,0]),0:len(field[0,:,0])]
        #    U = df.at[run_id,'Liquid Y-Velocity [m_per_y]'][:,:,8]
        #    V = df.at[run_id,'Liquid X-Velocity [m_per_y]'][:,:,8]
        #    plt.streamplot(Y, X, U, V, density=[2, 0.7])
        #    plt.xlabel("y")
        #    plt.ylabel("x")
        #elif view=="top":
        #    X, Y = np.mgrid[0:len(field[:,0,0]),0:len(field[0,:,0])]
        #    U = df.at[run_id,'Liquid Y-Velocity [m_per_y]'][:,:,-1]
        #    V = df.at[run_id,'Liquid X-Velocity [m_per_y]'][:,:,-1]
        #    plt.streamplot(Y, X, U, V, density=[2, 0.7])
        #    plt.xlabel("y")
        #    plt.ylabel("x")
        #elif view=="topish":
        #    X, Y = np.mgrid[0:len(field[:,0,0]),0:len(field[0,:,0])]
        #    U = df.at[run_id,'Liquid Y-Velocity [m_per_y]'][:,:,-3]
        #    V = df.at[run_id,'Liquid X-Velocity [m_per_y]'][:,:,-3]
        #    plt.streamplot(Y, X, U, V, density=[2, 0.7])
        #    plt.xlabel("y")
        #    plt.ylabel("x")
        #    
        #plt.show()

    pic_file_name = "visualization/pics/plot_phys_props"
    if plot_streamlines:
        pic_file_name += "_with_streamlines"
    if run_id:
        pic_file_name += "_RUN_" + str(run_id)
    pic_file_name += "_VIEW_" + view + ".jpg"
    print(f"Resulting picture is at {pic_file_name}")
    plt.savefig(pic_file_name)

# def slice_y(y, property):
#     property = 0 if property == "temperature" else property #1
#     return y.detach().numpy()[0,property,:,:]

def plot_y(data, name_pic="plot_y_exemplary"):
    n_subplots = len(data)
    fig, axes = plt.subplots(n_subplots,1,sharex=True,figsize=(20,3*(n_subplots)))
    #plt.title("Exemplary Comparison Input Output")
    
    for index, data_point in enumerate(data):
        plt.sca(axes[index])
        plt.imshow(data_point["data"].T)
        plt.gca().invert_yaxis()

        plt.xlabel("y")
        plt.ylabel("z")
        aligned_colorbar(label=data_point["property"])
    
    pic_file_name = f"runs/{name_pic}.jpg"
    print(f"Resulting picture is at {pic_file_name}")  
    plt.savefig(pic_file_name)
    
def plot_exemplary_learned_result(model, dataloaders, name_pic="plot_y_exemplary"):
    """not pretty but functional to get a first glimpse of how y_out looks compared to y_truth"""

    for data in dataloaders["train"]:
        x_exemplary = data["x"].float()
        y_true_exemplary = data["y"].float()
        y_out_exemplary = model(x_exemplary)
        # writer.add_image("y_out_test_0", y_out_test[0,0,:,:], dataformats="WH", global_step=0)
        # writer.add_image("y_true_test_0", y_true_test[0,0,:,:], dataformats="WH", global_step=0)
        # 
        # mse_loss = loss_fn(y_out_test, y_true_test)
        # loss = mse_loss
        # 
        # writer.add_scalar("loss", loss.item(), 0)
        break

    def make_dict(data, property, index):
        dict = {"data" : data, "property" : property}
        dict["data"] = dict["data"].detach().numpy()[0,index,:,:]
        # always uses first batch element
        # if property == "temperature" or property == "y-velocity":
        #     #dict["index"] = 0
        # elif property == "z-velocity":
        #     dict["data"] = dict["data"].detach().numpy()[0,1,:,:]
        #     #dict["index"] = 1
        # elif property == "pressure":
        #     dict["data"] = dict["data"].detach().numpy()[0,2,:,:]
        #     #dict["index"] = 2
        # elif property == "hp location":
        #     dict["data"] = dict["data"].detach().numpy()[0,3,:,:]
        #     #dict["index"] = 3
        # elif property == "init temperature":
        #     dict["data"] = dict["data"].detach().numpy()[0,4,:,:]
        #     #dict["index"] = 4
        return dict

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
    error_abs = torch.abs(error)
    list_to_plot = [
        make_dict(y_true_exemplary, "temperature true", 0),
        make_dict(y_out_exemplary, "temperature out", 0),
        make_dict(error, "error", 0),
        #make_dict(error_abs, "error abs", 0),
        make_dict(x_exemplary, "pressure", 0),
        make_dict(x_exemplary, "hp location", 1),
    ]

    plot_y(list_to_plot, name_pic=name_pic)

    return error


## helper to find inlet, outlet alias max and min Material_ID
# print(np.where(data["Material_ID"]==np.max(data["Material_ID"])))