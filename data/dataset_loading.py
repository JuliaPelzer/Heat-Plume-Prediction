from data.dataset import DatasetSimulationData, SettingsDataset
from data.dataloader import DataLoader
from data.transforms import NormalizeTransform, ComposeTransform, ReduceTo2DTransform, PowerOfTwoTransform, ToTensorTransform, SignedDistanceTransform
import os
from shutil import copyfile
import logging
from typing import List
import numpy as np
from data.utils import SettingsTraining

def init_data(settings: SettingsTraining, batch_size: int = 1000, labels: str = "t",):
    """Initialize dataset and dataloader for training."""
    
    reduce_to_2D: bool = True
    reduce_to_2D_xy: bool = True
    _do_assertions([reduce_to_2D, reduce_to_2D_xy], [batch_size], [settings.dataset_name, settings.path_to_datasets])

    # Prepare variables
    overfit: bool = False
    if not overfit:
        split = {'train': 0.7, 'val': 0.2, 'test': 0.1}
    else:
        split = {'train': 1, 'val': 0, 'test': 0}

    modes = ['train', 'val', 'test'] if split["test"] != 0 else ['train', 'val']
    split = _split_test_data_extra_folder(split, settings.path_to_datasets, settings.dataset_name)
    transforms = _build_transforms(reduce_to_2D, reduce_to_2D_xy)
    input_vars = _build_property_list(settings.inputs) 
    output_vars = _build_property_list(labels)

    # Init datasets
    datasets = {}
    means_stds_train_tuple=None
    for mode in modes:
        if mode == "test":
            dataset_name_temp = settings.dataset_name+"_TEST"
        else:
            dataset_name_temp = settings.dataset_name
        settings_dataset_temp = SettingsDataset(dataset_name=dataset_name_temp, input_vars_names=input_vars, output_vars_names=output_vars, mode=mode, split=split)
        temp_dataset = DatasetSimulationData(settings, settings_dataset_temp, transforms, means_stds_train_tuple,)
        
        if mode == "train":    
            means_stds_train_tuple = temp_dataset.mean_inputs, temp_dataset.std_inputs, temp_dataset.mean_labels, temp_dataset.std_labels
        datasets[mode] = temp_dataset

    # Init dataloaders
    dataloaders = {}
    for mode in modes:
        temp_dataloader = DataLoader(dataset=datasets[mode], batch_size=batch_size, shuffle=True, drop_last=False,)
        dataloaders[mode] = temp_dataloader

    # Logging
    len_datasets = ""
    for mode in modes:
        len_datasets += f"{mode}: {len(datasets[mode])} "
    logging.warning(f'init done [total number of datapoints/runs: {np.sum([len(datasets[mode]) for mode in modes])}], with {len_datasets}')
    return datasets, dataloaders

def make_dataset_for_test(settings: SettingsTraining):
    """Initialize dataset and dataloader for testing."""

    mode = "test"
    split = {'train': 0, 'val': 0, 'test': 1}
    labels = "t"

    transforms = _build_transforms(reduce_to_2D=True, reduce_to_2D_xy=True)
    input_vars = _build_property_list(settings.inputs) 
    output_vars = _build_property_list(labels)
    
    settings_dataset_temp = SettingsDataset(dataset_name=settings.dataset_name, input_vars_names=input_vars, output_vars_names=output_vars, mode=mode, split=split)
    dataset = DatasetSimulationData(settings, settings_dataset_temp, transforms)
    dataloader = DataLoader(dataset=dataset, batch_size=100, shuffle=True, drop_last=False,)
    logging.warning(f'init done [total number of datapoints: {len(dataset)}]')
    return dataset, dataloader


def _do_assertions(list_of_bools: List[bool], list_of_ints: List[int], list_of_strs: List[str]):
    for bool_ in list_of_bools:
        assert isinstance(bool_, bool), f"input parameters have to be bool, {bool_} is not"
    for int_ in list_of_ints:
        assert isinstance(int_, int), f"input parameters have to be int, {int_} is not"
    for str_ in list_of_strs:
        assert isinstance(str_, str), f"input parameters have to be str, {str_} is not"

def _build_transforms(reduce_to_2D: bool, reduce_to_2D_xy: bool):
    transforms_list = [ToTensorTransform()] #, PowerOfTwoTransform(oriented="left")]

    if reduce_to_2D: 
        transforms_list.append(ReduceTo2DTransform(reduce_to_2D_xy=reduce_to_2D_xy))
    transforms_list.append(SignedDistanceTransform())
    transforms_list.append(NormalizeTransform())

    transforms = ComposeTransform(transforms_list)
    return transforms

def _build_property_list(properties:str) -> List:
    vars_list = [properties[i] for i in range(len(properties))]
    for i in vars_list:
        assert i in ["x", "y", "z", "p", "t", "k", "i", "s"], "input parameters have to be a string of characters, each of which is either x, y, z, p, t, k, i, s"

    vars = []
    if 'x' in vars_list:
        vars.append("Liquid X-Velocity [m_per_y]")
    if 'y' in vars_list:
        vars.append("Liquid Y-Velocity [m_per_y]")
    if 'z' in vars_list:
        vars.append("Liquid Z-Velocity [m_per_y]")
    if 'p' in vars_list:
        vars.append("Liquid Pressure [Pa]") # unstructured grid, else Liquid_Pressure [Pa]
    if 't' in vars_list:
        vars.append("Temperature [C]")
    if 'k' in vars_list:
        vars.append("Permeability X [m^2]")
    if 'i' in vars_list:
        vars.append("Material ID") # unstructured grid, else Material_ID
    if 's' in vars_list:
        vars.append("SDF")

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
