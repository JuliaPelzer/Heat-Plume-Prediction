from data.dataset import DatasetSimulationData
from data.dataloader import DataLoader
from data.transforms import NormalizeTransform, ComposeTransform, ReduceTo2DTransform, PowerOfTwoTransform, ToTensorTransform
import logging
from typing import List
import yaml

def init_data(reduce_to_2D: bool = True, reduce_to_2D_xy: bool = False, overfit: bool = False, normalize: bool = True, 
              just_plotting: bool = False, batch_size: int = 100, inputs: str = "xyzpt", labels: str = "txyz",
              dataset_name: str = "approach2_dataset_generation_simplified/dataset_HDF5_testtest", 
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

    datasets = {}
    transforms_list = [
        ToTensorTransform(), PowerOfTwoTransform(oriented="left")]
    if reduce_to_2D:
        transforms_list.append(ReduceTo2DTransform(reduce_to_2D_xy=reduce_to_2D_xy))
    if normalize:
        transforms_list.append(NormalizeTransform(reduced_to_2D=reduce_to_2D))
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

    input_vars = _build_property_list(inputs)
    input_vars.append("Material_ID") # if structured grid
    input_vars.append("Material ID") # if unstructured grid
    output_vars = _build_property_list(labels)
    
    # read json file for dimensions
    with open(f"{path_to_datasets}/{dataset_name}/inputs/settings_perm_field.yaml", "r") as f:
        perm_settings = yaml.safe_load(f)
    dimensions_of_datapoint = perm_settings["ncells"]

    for mode in ['train', 'val', 'test']:
        temp_dataset = DatasetSimulationData(
            dataset_name=dataset_name, dataset_path=path_to_datasets,
            transform=transforms, input_vars_names=input_vars,
            output_vars_names=output_vars,  # . "Liquid_Pressure [Pa]"
            mode=mode, split=split, dimensions_of_datapoint=dimensions_of_datapoint,
        )
        datasets[mode] = temp_dataset

    # Create a dataloader for each split.
    dataloaders = {}
    for mode in ['train', 'val', 'test']:
        temp_dataloader = DataLoader(
            dataset=datasets[mode],
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )
        dataloaders[mode] = temp_dataloader
    print(f'init done [total number of datapoints/runs: {len(datasets["train"])+len(datasets["val"])+len(datasets["test"])}], with train: {len(datasets["train"])}, val: {len(datasets["val"])}, test: {len(datasets["test"])}')

    return datasets, dataloaders


def _build_property_list(properties:str) -> List:
    vars_list = [properties[i] for i in range(len(properties))]
    for i in vars_list:
        assert i in ["x", "y", "z", "p", "t", "k"], "input parameter inputs has to be a string of characters, each of which is either x, y, z, p, t, k"

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