import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   #if like me you do not have a lot of memory in your GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "" #then these two lines force keras to use your CPU
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py
from tqdm.auto import tqdm

## read data from source folder into dataframe
def read_data_df(path_dataset):
    time_init =     "Time:  0.00000E+00 y"
    time_final =    "Time:  5.00000E+00 y"
    names_of_runs = []
    names_of_properties = []
    data = []

    for file in tqdm(list(os.listdir(path_dataset))):
        try:
            path_data = os.path.join(path_dataset,file)
            if not os.path.isdir(path_data):
                continue
            #print(file)
            names_of_runs.append(file)
            filename = os.path.join(path_data,"pflotran.h5") 
            interim_array = []
            with h5py.File(filename, "r") as f:
                for key, value in f[time_final].items():
                    if key=='Liquid_Pressure [Pa]':
                        interim_array.append(np.array(f[time_init]["Liquid_Pressure [Pa]"]))
                    else:
                        interim_array.append(np.array(value))
                names_of_properties = list(f[time_final].keys())
            data.append(interim_array)

        except Exception as e:
            tqdm.write(f"lololololololoo: {e}")
    
    df = pd.DataFrame(data=data, index=names_of_runs, columns=names_of_properties)

    return df

## helper function for plotting
def aligned_colorbar(*args,**kwargs):
    cax = make_axes_locatable(plt.gca()).append_axes("right",size= 0.3,pad= 0.05)
    plt.colorbar(*args,cax=cax,**kwargs)
    
## plotting function: all physical properties, depending on the view also with streamlines
def plot_sample_df(df, run_id, view="top"):
    n_dims = len(df.columns)
    fig, axes = plt.subplots(n_dims+1,1,sharex=True,figsize=(20,3*(n_dims+1)))
    plt.figure(figsize= (20,3*(n_dims+1)))
    for column, (i) in zip(df.columns, range(n_dims)):
        plt.sca(axes[i])
        field = df.at[run_id, column]
        if len(field.shape) != 3:
            # no 3D data
            continue
        index = column.find(' [')
        title = column
        if index != -1:
            title = column[:index]
        plt.title(title)
        if view=="side":
            plt.imshow(field[11,:,::-1].T)
            plt.xlabel("y")
            plt.ylabel("z")
        elif view=="side_hp":
            plt.imshow(field[8,:,::-1].T)
            plt.xlabel("y")
            plt.ylabel("z")
        elif view=="top_hp":
            plt.imshow(field[:,:,8])
            plt.xlabel("y")
            plt.ylabel("x")
        elif view=="top":
            plt.imshow(field[:,:,-1])
            plt.xlabel("y")
            plt.ylabel("x")
        elif view=="topish":
            plt.imshow(field[:,:,-3])
            plt.xlabel("y")
            plt.ylabel("x")
        aligned_colorbar(label=column)
        
    #streamlines
    plt.sca(axes[i+1])
    plt.title("Streamlines")
    if view=="side":
        Z, Y = np.mgrid[0:len(field[0,0,:]),0:len(field[0,:,0])]
        U = df.at[run_id,'Liquid Y-Velocity [m_per_y]'][11,:,::-1]
        V = df.at[run_id,'Liquid Z-Velocity [m_per_y]'][11,:,::-1]
        plt.streamplot(Y, Z, U.T, V.T, density=[2, 0.7])
        plt.xlabel("y")
        plt.ylabel("z")
    elif view=="side_hp":
        Z, Y = np.mgrid[0:len(field[0,0,:]),0:len(field[0,:,0])]
        U = df.at[run_id,'Liquid Y-Velocity [m_per_y]'][8,:,::-1]
        V = df.at[run_id,'Liquid Z-Velocity [m_per_y]'][8,:,::-1]
        plt.streamplot(Y, Z, U.T, V.T, density=[2, 0.7])
        plt.xlabel("y")
        plt.ylabel("z")
    elif view=="top_hp":
        X, Y = np.mgrid[0:len(field[:,0,0]),0:len(field[0,:,0])]
        U = df.at[run_id,'Liquid Y-Velocity [m_per_y]'][:,:,8]
        V = df.at[run_id,'Liquid X-Velocity [m_per_y]'][:,:,8]
        plt.streamplot(Y, X, U, V, density=[2, 0.7])
        plt.xlabel("y")
        plt.ylabel("x")
    elif view=="top":
        X, Y = np.mgrid[0:len(field[:,0,0]),0:len(field[0,:,0])]
        U = df.at[run_id,'Liquid Y-Velocity [m_per_y]'][:,:,-1]
        V = df.at[run_id,'Liquid X-Velocity [m_per_y]'][:,:,-1]
        plt.streamplot(Y, X, U, V, density=[2, 0.7])
        plt.xlabel("y")
        plt.ylabel("x")
    elif view=="topish":
        X, Y = np.mgrid[0:len(field[:,0,0]),0:len(field[0,:,0])]
        U = df.at[run_id,'Liquid Y-Velocity [m_per_y]'][:,:,-3]
        V = df.at[run_id,'Liquid X-Velocity [m_per_y]'][:,:,-3]
        plt.streamplot(Y, X, U, V, density=[2, 0.7])
        plt.xlabel("y")
        plt.ylabel("x")
        
    #plt.show()
    plt.savefig(f"plot_phys_props_plus_streamlines_{run_id}_{view}.jpg")

## data cleaning: cut of edges - to get rid of problems with boundary conditions
def data_cleaning(data):
    for key, (value) in data.items():
        data[key] = data[key][:,1:-1,1:-3,1:-1]
        print(f"{key} :{np.shape(data[key])}")
    return data

def data_cleaning_df(df):
    for label, content in df.items():
        for index in range(len(content)):
            content[index] = content[index][1:-1,1:-3,1:-1]
    return df


if __name__=="__main__":
    path_dir = "/home/pelzerja/Development/simulation_groundtruth_pflotran/Phd_simulation_groundtruth/approach2_dataset_generation_simplified"
    dataset_name = "dataset_HDF5_test" #dataset_HDF5_uniformly_distributed_data
    path_dataset = os.path.join(path_dir, dataset_name)

    # read data from file
    df = read_data_df(path_dataset)

    # preprocessing
    df = data_cleaning_df(df)

    # visualize physical properties in one data point
    plot_sample_df(df,"RUN_1",view="side_hp")