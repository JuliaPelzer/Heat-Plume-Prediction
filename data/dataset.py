"""
Dataset Class
"""

import logging
from typing import List, Dict, Tuple, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from data.transforms import ComposeTransform
from data.utils import PhysicalVariables, DataPoint, _assertion_error_2d, get_dimensions, SettingsTraining
import os
import numpy as np
import h5py
import yaml
from torch import Tensor, square, mean, sqrt, zeros
from torch.utils.data import Dataset as TorchDataset


@dataclass
class SettingsDataset:
    dataset_name: str
    mode: str
    split: Dict[str, float]
    input_vars_names: List = field(default_factory=["Liquid X-Velocity [m_per_y]", "Liquid Y-Velocity [m_per_y]", "Liquid Z-Velocity [m_per_y]", "Liquid_Pressure [Pa]", "Material_ID", "Temperature [C]"])
    output_vars_names: List = field(default_factory=["Liquid_Pressure [Pa]", "Temperature [C]"])
    sdf: bool = True

    def __post_init__(self):
        assert self.mode in ["train", "val", "test"], "wrong mode for dataset given"

        split_values = [value for value in self.split.values()]
        assert np.round(sum(split_values)) == 1.0, "split values do not sum up to 1"

@dataclass
class Dataset(ABC):
    """
    Abstract Dataset Base Class
    All subclasses must define __getitem__() and __len__()
    """
    dataset_name:str
    dataset_path:str=None
    # Usually the dataset is stored where it is produced and just referred to but if no path is given, it is supposed to be in a neighbouring folder called datasets
    dataset_path = dataset_path if dataset_path else os.path.join(os.path.dirname(os.path.abspath(os.getcwd())), "datasets")
    # The actual archive name should be all the text of the url after the last '/'.

    @abstractmethod
    def __getitem__(self, index: int):
        """Return data sample at given index"""

    @abstractmethod
    def __len__(self):
        """Return size of the dataset"""

    
    def check_for_dataset(self) -> str:
        """
        Check if the dataset exists and is not empty.
        Dataset should be in the following folder self.dataset_path/<dataset_name>
        """
        dataset_path_full = os.path.join(self.dataset_path, self.dataset_name)
        if not os.path.exists(dataset_path_full):
            raise ValueError(f"Dataset {self.dataset_name} does not exist in {dataset_path_full}")
        if len(os.listdir(dataset_path_full)) == 0:
            raise ValueError(f"Dataset {self.dataset_name} is empty")
        return dataset_path_full

class DatasetSimulationData(TorchDataset, Dataset):
    """Groundwaterflow and heatpumps dataset class dataset has dimensions of NxCxHxWxD,
    where N is the number of runs/data points, C is the number of channels, HxWxD are the spatial dimensions

    Returns
    -------
    dataset of size dict(Nx dict(Cx value(HxWxD)))
    """
    def __init__(self, settings_general: SettingsTraining, settings_dataset: SettingsDataset, transform: ComposeTransform=None,
                 means_stds_train_tuple:Tuple[Dict, Dict, Dict, Dict]=None)-> Dataset:

        Dataset.__init__(self, dataset_name=settings_dataset.dataset_name, dataset_path=settings_general.path_to_datasets)
        
        self.settings_dataset = settings_dataset
        self.transform = transform
        self.dataset_path = self.check_for_dataset()
        
        self.data_paths, self.runs = self.make_dataset(self)
        logging.info(f"Dataset {self.dataset_name} in mode {self.settings_dataset.mode} has {len(self.data_paths)} runs, named {self.runs}")
        self.time_first =     "   0 Time  0.00000E+00 y"
        
        old_dataset=False
        if old_dataset:
            self.time_final =    "   2 Time  5.00000E+00 y" # other dataset
        else:
            self.time_final =    "   3 Time  5.00000E+00 y"
        
        self.datapoints = {}
        self.keep_datapoints_in_memory = True
        self.dimensions_of_datapoint:Tuple[int, int, int] = get_dimensions(f"{self.dataset_path}")
        # TODO put selection of input and output variables in a separate transform function (see ex4 - FeatureSelectorAndNormalizationTransform)
        self.input_vars_empty_value = PhysicalVariables(self.time_first, self.settings_dataset.input_vars_names) # collection of overall information but not std, mean
        self.output_vars_empty_value = PhysicalVariables(self.time_final, self.settings_dataset.output_vars_names)

        self.has_to_calc_mean_std = True
        if self.settings_dataset.mode=="train":
            self.mean_inputs, self.std_inputs, self.mean_labels, self.std_labels = self._calc_mean_std_dataset()   # calc mean, std for all dataset runs
            self.save_mean_std_dataset(self.mean_inputs, self.std_inputs, self.mean_labels, self.std_labels, os.path.join(os.getcwd(), "runs", settings_general.name_folder_destination))
        elif means_stds_train_tuple is not None:
            self.mean_inputs, self.std_inputs, self.mean_labels, self.std_labels = means_stds_train_tuple
        elif self.settings_dataset.mode=="test":
            self.mean_inputs, self.std_inputs, self.mean_labels, self.std_labels = self.load_mean_std_dataset(os.path.join(os.getcwd(), "runs", settings_general.name_folder_destination))
        else:
            raise ValueError("problem with means_stds_train_tuple(must be given for mode val)")
        self.has_to_calc_mean_std = False
        
    def __len__(self):
        # Return the length of the dataset (number of runs), but fill them only on their first call
        return len(self.runs)

    @staticmethod
    def make_dataset(self) -> Tuple[List[str], List[str]]:
        """
        Create the simulation dataset by preparing a list of samples
        Simulation data are sorted in an ascending order by run number
        :returns: (data_paths, runs) where:
            - data_paths is a list containing paths to all simulation runs in the dataset, NOT the actual simulated data
            - runs is a list containing one label per run
        """
        dataset_path = self.dataset_path
        set_data_paths_runs, runs = [], []
        found_dataset = False

        logging.info(f"Directory of currently used dataset is: {dataset_path}")
        for _, folders, _ in os.walk(dataset_path):
            for folder in folders:
                for file in os.listdir(os.path.join(dataset_path, folder)):
                    if file == "pflotran.h5":
                        set_data_paths_runs.append((folder, os.path.join(dataset_path, folder, file)))
                        found_dataset = True
        # Sort the data and runs in ascending order
        set_data_paths_runs = sorted(set_data_paths_runs, key=lambda val: int(val[0].strip('RUN_')))
        runs = [data_path[0] for data_path in set_data_paths_runs]
        data_paths = [data_path[1] for data_path in set_data_paths_runs]
        # Split the data and runs into train, val
        if not self.settings_dataset.mode == "test":
            data_paths, runs = self._select_split(data_paths, runs)
        if not found_dataset:
            raise ValueError("No dataset found")
        assert len(data_paths) == len(runs)

        return data_paths, runs

    def get_input_properties(self) -> List[str]:
        return list(self.input_vars_empty_value.keys)

    def get_output_properties(self) -> List[str]:
        return list(self.output_vars_empty_value.keys)
        
    def __getitem__(self, index:int) -> Dict[int, DataPoint]: 
        """
        Get a data point as a dict at a given run id (index) in the dataset
        If the datapoint was loaded before, it is called only, if not: it is initialized with calling transform etc.
        
        Parameters
        ----------
        index : int
            Run_id of the data point to be loaded (check in list of paths)

        Returns
        -------
        datapoint : dict
            Dictionary containing the data point with the following keys:
                - inputs: dict
                    Dictionary containing the input data with the following keys:
                        - time: str
                            Time of the data point
                        - properties:  
                            List of the properties of the data point
                        - values: torch.Tensor 
                            Tensor of the values of the data point
                - labels: dict
                    Dictionary containing the output data with the same keys as the input data
        """
        # TODO x is a numpy array of shape CxHxWxD (C=channels, HxWxD=spatial dimensions)
        # TODO y is a numpy array of shape CxHxWxD (C= output channels, HxWxD=spatial dimensions)
        if self.keep_datapoints_in_memory:
            if index not in self.datapoints.keys():
                self.datapoints[index] = self.load_datapoint(index)
            return self.datapoints[index]
        else:
            return self.load_datapoint(index)

    def load_datapoint(self, index:int) -> DataPoint:
        """
        Load a datapoint at a given index (from data_paths) in the dataset with all input- and label-data
        and applies transforms if possible/ if any where given
        checks if the data is 2D or 3D - else: error
        """
        datapoint = DataPoint(index)

        datapoint.inputs = self._load_data_as_numpy(self.data_paths[index], self.input_vars_empty_value)
        datapoint.labels = self._load_data_as_numpy(self.data_paths[index], self.output_vars_empty_value)
        #try:
        loc_hp = datapoint.get_loc_hp()
        if self.has_to_calc_mean_std:
            # mean, std first have to be calculated for later use
            datapoint.inputs = self.transform(datapoint.inputs, loc_hp=loc_hp)
            datapoint.labels = self.transform(datapoint.labels, loc_hp=loc_hp)
        else:
            datapoint.inputs = self.transform(datapoint.inputs, loc_hp=loc_hp, mean_val=self.mean_inputs, std_val=self.std_inputs)
            datapoint.labels = self.transform(datapoint.labels, loc_hp=loc_hp, mean_val=self.mean_labels, std_val=self.std_labels)

        _assertion_error_2d(datapoint)
        return datapoint

    def reverse_transform(self, datapoint:Union[DataPoint, Tensor]):
        """
        Reverse the transformation of the data.
        """

        if self.settings_dataset.sdf:
            props_to_exclude_from_norm = ["Material ID", "Material_ID"]
        else:
            props_to_exclude_from_norm = []

        if isinstance(datapoint, Tensor):
            for id, property in enumerate(self.mean_inputs.keys()):
                if property not in props_to_exclude_from_norm:
                    datapoint[:,id] = self.transform.reverse_tensor_input(datapoint[:,id], mean_val=self.mean_inputs[property], std_val=self.std_inputs[property])
            return datapoint
        else:
            # MaterialID is excluded later (in transform-normalize)
            datapoint.inputs = self.transform.reverse(datapoint.inputs, mean_val=self.mean_inputs, std_val=self.std_inputs)
            datapoint.labels = self.transform.reverse(datapoint.labels, mean_val=self.mean_labels, std_val=self.std_labels)

        _assertion_error_2d(datapoint)

        return datapoint

    def reverse_transform_temperature(self, temperature:Tensor) -> Tensor:
        temperature = self.transform.reverse_tensor_input(temperature, mean_val=self.mean_labels["Temperature [C]"], std_val=self.std_labels["Temperature [C]"])
        return temperature

    def _select_split(self, data_paths:List[str], labels:List[str]) -> Tuple[List[str], List[str]]:
        """
        Depending on the mode of the dataset, deterministically split it.
        
        Parameters
        ----------
        data_paths: list containing paths to all data_points in the dataset
        labels: list containing one label/RUN_xx per data_point
        
        Returns
        -------
        data_paths: where only the indices for the corresponding data split are selected
        runs: where only the indices for the corresponding data split are selected
        """
        " !!only done for train and val, not for test!!"

        fraction_train = self.settings_dataset.split['train']
        fraction_val = self.settings_dataset.split['val']
        num_samples = len(data_paths)
        num_train = int(np.round(num_samples * fraction_train, 0))
        num_valid = int(np.round(num_samples * fraction_val, 0))
            
        np.random.seed(0)   
        # TODO have to keep it somehow or split "RUN_x" beforehand - to make sure that validation is actually working on other files than training!! remove later only for testing
        # TODO check communicate rand_perm with the datasets of all 3 modes?
        rand_perm = np.random.permutation(num_samples)
        
        if self.settings_dataset.mode == 'train':
            indices = rand_perm[:num_train]
        elif self.settings_dataset.mode == 'val':
            indices = rand_perm[num_train:num_train+num_valid]

        if isinstance(data_paths, list): 
            return list(np.array(data_paths)[indices]), list(np.array(labels)[indices])
        else: 
            logging.error("I did not expect this to be the case at any moment!")
            return data_paths[indices], list(np.array(labels)[indices])
        
    def _load_data_as_numpy(self, data_path:str, variables:PhysicalVariables) -> PhysicalVariables:
        """
        Load data from h5 file on data_path, but only the variables named in variables.get_ids() at time stamp variables.time
        Sets the values of each PhysicalVariable in variables to the loaded data.
        """
        loaded_datapoint = PhysicalVariables(variables.time)
        with h5py.File(data_path, "r") as file:
            for key, value in file[variables.time].items():
                if key in variables.get_ids_list(): # properties
                    loaded_datapoint[key] = np.array(value).reshape(self.dimensions_of_datapoint, order='F')
        return loaded_datapoint

    def _calc_mean_std_dataset(self):
        """
        Calculate mean and std of the dataset.
        """
        mean_in = {}
        mean_labels = {}
        number_datapoints = len(self)

        if self.settings_dataset.sdf:
            props_to_exclude_from_norm = ["Material ID", "Material_ID"]
        else:
            props_to_exclude_from_norm = []
        # load all dataset + calc mean
        for run in range(len(self)):
            for key, value in self[run].inputs.items():
                if key not in props_to_exclude_from_norm:
                    mean_in[key] = mean_in[key]+value.mean_orig if key in mean_in.keys() else value.mean_orig
                else: 
                    # for MaterialID because later needed in dataset:reverse_transform for order between other properties
                    mean_in[key] = zeros(1)
            for key, value in self[run].labels.items():
                mean_labels[key] = mean_labels[key]+value.mean_orig if key in mean_labels.keys() else value.mean_orig

        for key, prop in mean_in.items(): prop /= number_datapoints
        for key, prop in mean_labels.items(): prop /= number_datapoints

        # calc std
        std_in = {}
        std_labels = {}
        squaresum_mean_in = {}
        squaresum_mean_labels = {}
        for run in range(len(self)):
            for key, value in self[run].inputs.items():
                if key not in props_to_exclude_from_norm:
                    squaresum_mean_in[key] = squaresum_mean_in[key] + square(value.value-mean_in[key]) if key in std_in.keys() else square(value.value-mean_in[key])
                else: 
                    # for MaterialID because later needed in dataset:reverse_transform for order between other properties
                    squaresum_mean_in[key] = zeros(1)
            for key, value in self[run].labels.items():
                squaresum_mean_labels[key] = squaresum_mean_labels[key] + square(value.value-mean_labels[key]) if key in std_labels.keys() else square(value.value-mean_labels[key])
        
        for key, prop in squaresum_mean_in.items(): std_in[key]=sqrt(mean(prop))
        for key, prop in squaresum_mean_labels.items(): std_labels[key]=sqrt(mean(prop))

        self.datapoints = {}    # del all datapoints created in this process to free space
        return mean_in, std_in, mean_labels, std_labels

    def save_mean_std_dataset(self, mean_in, std_in, mean_labels, std_labels, path_mean_std:str):
        """
        Save mean and std of the dataset.
        """
        name_file = "means_stds_train_dataset.yaml"
        for prop in mean_in.keys():
            mean_in[prop] = mean_in[prop].numpy().tolist()
            std_in[prop] = std_in[prop].numpy().tolist()
        for prop in mean_labels.keys():
            mean_labels[prop] = mean_labels[prop].numpy().tolist()
            std_labels[prop] = std_labels[prop].numpy().tolist()

        with open(os.path.join(path_mean_std, name_file), 'w') as f:
            yaml.dump({"mean_inputs": mean_in, "std_inputs": std_in, "mean_labels": mean_labels, "std_labels": std_labels}, f, default_flow_style=False)

    def load_mean_std_dataset(self, path_mean_std:str):
        """
        Load mean and std of the dataset.
        """
        name_file = "means_stds_train_dataset.yaml"
        with open(os.path.join(path_mean_std, name_file), 'r') as f:
            means_stds = yaml.safe_load(f)
        return means_stds["mean_inputs"], means_stds["std_inputs"], means_stds["mean_labels"], means_stds["std_labels"]