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

# TODO: look at vispy library for plotting 3D data

## helper function for plotting
def aligned_colorbar(*args,**kwargs):
    cax = make_axes_locatable(plt.gca()).append_axes("right",size= 0.3,pad= 0.05)
    plt.colorbar(*args,cax=cax,**kwargs)
    
def plot_sample(dataset : GWF_HP_Dataset, run_id : int, view="top") -> None:
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

    n_dims = len(dataset.get_input_properties()) + len(dataset.get_output_properties())
    data = dataset[run_id]
    index_overall = 0
    fig, axes = plt.subplots(n_dims+1,1,sharex=True,figsize=(20,3*(n_dims+1)))

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
            "cut_slice": np.s_[:,:,8],
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
            "cut_slice": np.s_[11,:,::-1],
            "transpose" : True
        },
        "side_hp": {
            "x_label": "y",
            "y_label": "z",
            "cut_slice": np.s_[8,:,::-1],
            "transpose" : True
        }
    }

    def plot_properties(data : np.ndarray, property_names : List[str]) -> None:
        """
        Plot all properties of one data point

        Parameters
        ----------
            data : np.ndarray
                Datapoint to plot, dimensions: channels x HxWxD
            property_names : List[str]
                List of all properties to plot

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
            title = property_names[channel]
            if index != -1:
                title = property_names[channel][:index]
            plt.title(title)
            # plot field, if view_dict transpose is true, transpose the field
            if view_dict[view]["transpose"]:
                plt.imshow(field[view_dict[view]["cut_slice"]].T)
            else:
                plt.imshow(field[view_dict[view]["cut_slice"]])
            
            plt.xlabel(view_dict[view]["x_label"])
            plt.ylabel(view_dict[view]["y_label"])

            aligned_colorbar(label=property_names[channel])
            index_overall += 1
        
    plot_properties(data['x'], property_names=dataset.get_input_properties())
    plot_properties(data['y'], property_names=dataset.get_output_properties())

    #streamlines
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
    plt.savefig(f"plot_phys_props_plus_streamlines_{run_id}_{view}.jpg")
## data cleaning: cut of edges - to get rid of problems with boundary conditions
def data_cleaning_df(df):
    for label, content in df.items():
        for index in range(len(content)):
            content[index] = content[index][1:-1,1:-3,1:-1]
    return df
## helper to find inlet, outlet alias max and min Material_ID
# print(np.where(data["Material_ID"]==np.max(data["Material_ID"])))

# main function combining all reading and visualizing of input
def read_and_visualize_data_as_df(dataset_name, input_vars, plot_bool=False, run_id="RUN_0", view_id="side_hp"):
    path_dir = "/home/pelzerja/Development/simulation_groundtruth_pflotran/Phd_simulation_groundtruth/approach2_dataset_generation_simplified"
    path_dataset = os.path.join(path_dir, dataset_name)

    # read data from file
    df = read_data_df(path_dataset, input_vars)

    # preprocessing
    df = data_cleaning_df(df)

    # visualize physical properties in one data point
    if plot_bool:
        plot_sample_df(df,run_id,view=view_id)

    return df


if __name__=="__main__":
    df = read_and_visualize_data_as_df(dataset_name = "dataset_HDF5_testtest", input_vars=["Liquid X-Velocity [m_per_y]", "Liquid Y-Velocity [m_per_y]",
       "Liquid Z-Velocity [m_per_y]"], plot_bool=True, run_id = "RUN_1", view_id="side_hp")