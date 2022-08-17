import learn_process as lp
import visualization.visualize_data as vis
import sys
import logging

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
    vis.plot_data_inner(tensor_dict, prop_in, prop_out, view=view, oriented="left")


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
