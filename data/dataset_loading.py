from data.dataset import DatasetSimulationData
from data.dataloader import DataLoader
from data.transforms import NormalizeTransform, ComposeTransform, ReduceTo2DTransform, PowerOfTwoTransform, ToTensorTransform
import os
from shutil import copyfile
import logging
from typing import List
import numpy as np

def init_data(reduce_to_2D: bool = True, reduce_to_2D_xy: bool = False, overfit: bool = False, normalize: bool = True, 
              just_plotting: bool = False, batch_size: int = 1000, inputs: str = "xyzpt", labels: str = "txyz",
              dataset_name: str = "perm_pressure1D_10dp", 
              path_to_datasets: str = "/home/pelzerja/Development/simulation_groundtruth_pflotran/Phd_simulation_groundtruth/datasets"):
    """
    Initialize dataset and dataloader for training.

    Parameters
    ----------
    reduce_to_2D : If true, reduce the dataset to 2D instead of 3D
    overfit : If true, only use a small subset of the dataset for training, achieved by cheating: changing the split ratio
    normalize : If true, normalize the dataset, usually the case; not true for testing input data magnitudes etc 
    dataset_name : Name of the dataset to use (has to be the same as in the folder)
    batch_size : Size of the batch to use for training

    Returns
    -------
    datasets : dict
        Dictionary of datasets, with keys "train", "val", "test"
    dataloaders : dict
        Dictionary of dataloaders, with keys "train", "val", "test"
    """
    assert isinstance(reduce_to_2D, bool) and isinstance(reduce_to_2D_xy, bool) and isinstance(overfit, bool) and isinstance(normalize, bool) and isinstance(
        just_plotting, bool), "input parameters reduce_to_2D, reduce_to_2D_wrong, overfit, normalize, just_plotting have to be bool"
    assert isinstance(
        batch_size, int), "input parameter batch_size has to be int"
    assert isinstance(dataset_name, str) and isinstance(
        path_to_datasets, str), "input parameters dataset_name, path_to_datasets have to be str"

    transforms_list = [ToTensorTransform(), PowerOfTwoTransform(oriented="left")]
    if reduce_to_2D:
        transforms_list.append(ReduceTo2DTransform(reduce_to_2D_xy=reduce_to_2D_xy))
    if normalize:
        transforms_list.append(NormalizeTransform())

    logging.info(f"transforms_list: {transforms_list}")

    transforms = ComposeTransform(transforms_list)
    if not overfit:
        split = {'train': 0.7, 'val': 0.2, 'test': 0.1}
    else:
        split = {'train': 1, 'val': 0, 'test': 0}

    # just plotting (for Marius)
    if just_plotting:
        split = {'train': 1, 'val': 0, 'test': 0}
        transforms = None
    
    modes = ['train', 'val', 'test'] if split["test"] != 0 else ['train', 'val']
    split = _split_test_data_extra_folder(split, path_to_datasets, dataset_name)

    input_vars = _build_property_list(inputs)
    input_vars.append("Material_ID") # if structured grid
    input_vars.append("Material ID") # if unstructured grid
    output_vars = _build_property_list(labels)

    datasets = {}
    means_stds_train_tuple=None
    for mode in modes:
        dataset_name_temp = dataset_name+"_TEST" if mode=="test" else dataset_name

        temp_dataset = DatasetSimulationData(
            dataset_name=dataset_name_temp, dataset_path=path_to_datasets,
            transform=transforms, input_vars_names=input_vars,
            output_vars_names=output_vars,
            mode=mode, split=split, normalize_bool=normalize,
            means_stds_train_tuple=means_stds_train_tuple
        )
        if mode == "train" and normalize:    
            means_stds_train_tuple = temp_dataset.mean_inputs, temp_dataset.std_inputs, temp_dataset.mean_labels, temp_dataset.std_labels
        datasets[mode] = temp_dataset

    # Create a dataloader for each split.
    dataloaders = {}
    for mode in modes:
        temp_dataloader = DataLoader(
            dataset=datasets[mode],
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )
        dataloaders[mode] = temp_dataloader

    len_datasets = ""
    for mode in modes:
        len_datasets += f"{mode}: {len(datasets[mode])} "
    print(f'init done [total number of datapoints/runs: {np.sum([len(datasets[mode]) for mode in modes])}], with {len_datasets}')
    return datasets, dataloaders


def _build_property_list(properties:str) -> List:
    vars_list = [properties[i] for i in range(len(properties))]
    for i in vars_list:
        assert i in ["x", "y", "z", "p", "t", "k"], "input parameters have to be a string of characters, each of which is either x, y, z, p, t, k"

    vars = []
    if 'x' in vars_list:
        vars.append("Liquid X-Velocity [m_per_y]")
    if 'y' in vars_list:
        vars.append("Liquid Y-Velocity [m_per_y]")
    if 'z' in vars_list:
        vars.append("Liquid Z-Velocity [m_per_y]")
    if 'p' in vars_list:
        vars.append("Liquid_Pressure [Pa]") # if structured grid
        vars.append("Liquid Pressure [Pa]") # if unstructured grid
    if 't' in vars_list:
        vars.append("Temperature [C]")
    if 'k' in vars_list:
        vars.append("Permeability X [m^2]")

    return vars

def _split_test_data_extra_folder(split:dict, path_to_datasets:str, dataset_name:str):
    if split["test"] != 0:
        # if no test folder yet, create one
        if not os.path.exists(f"{path_to_datasets}/{dataset_name}_TEST"):
            logging.warning("No TEST data folder yet, creating one now")
            os.mkdir(f"{path_to_datasets}/{dataset_name}_TEST")
            number_datapoints = len(os.listdir(f"{path_to_datasets}/{dataset_name}"))-1
            number_test_files = int(number_datapoints * split["test"])
            
            # select and move random datapoints/ files to test folder
            indices_test = np.random.permutation(number_datapoints)[:number_test_files]
            for index_file in indices_test:
                os.rename(f"{path_to_datasets}/{dataset_name}/RUN_{index_file}", f"{path_to_datasets}/{dataset_name}_TEST/RUN_{index_file}")

            # copy inputs folder
            os.mkdir(f"{path_to_datasets}/{dataset_name}_TEST/inputs")
            for file in os.listdir(f"{path_to_datasets}/{dataset_name}/inputs"):
                copyfile(f"{path_to_datasets}/{dataset_name}/inputs/{file}", f"{path_to_datasets}/{dataset_name}_TEST/inputs/{file}")
                
        # if test folder already exists, use it
        else:
            logging.warning("Test data folder exists already, using it")

        split["train"] = np.round(split["train"]/(1-split["test"]), 2)
        split["val"] = np.round(1-split["train"], 2)
        split["test"] = 0

    return split
