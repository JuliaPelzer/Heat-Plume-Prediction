"""
Dataset Class
"""

import logging
from typing import List, Dict, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from data.transforms import ComposeTransform
from data.utils import PhysicalVariables, DataPoint, _assertion_error_2d, get_dimensions
import os, sys
import numpy as np
import h5py
from torch import Tensor

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
            raise ValueError(f"Dataset {self.dataset_name} does not exist")
        if len(os.listdir(dataset_path_full)) == 0:
            raise ValueError(f"Dataset {self.dataset_name} is empty")
        return dataset_path_full

class DatasetSimulationData(Dataset):
    """Groundwaterflow and heatpumps dataset class dataset has dimensions of NxCxHxWxD,
    where N is the number of runs/data points, C is the number of channels, HxWxD are the spatial dimensions

    Returns
    -------
    dataset of size dict(Nx dict(Cx value(HxWxD)))
    """
    def __init__(self, dataset_name:str="approach2_dataset_generation_simplified/dataset_HDF5_testtest",
                 dataset_path:str="/home/pelzerja/Development/simulation_groundtruth_pflotran/Phd_simulation_groundtruth",
                 transform:ComposeTransform=None,
                 mode:str="train",
                 split:Dict[str, float]={'train': 0.6, 'val': 0.2, 'test': 0.2},
                 input_vars_names:List[str]=["Liquid X-Velocity [m_per_y]", "Liquid Y-Velocity [m_per_y]", "Liquid Z-Velocity [m_per_y]", 
                 "Liquid_Pressure [Pa]", "Material_ID", "Temperature [C]"], # "hp_power"
                 output_vars_names:List[str]=["Liquid_Pressure [Pa]", "Temperature [C]"],
                 **kwargs)-> Dataset:
        super().__init__(dataset_name=dataset_name, dataset_path=dataset_path, **kwargs)
        assert mode in ["train", "val", "test"], "wrong mode for dataset given"
        
        self.mode = mode
        self.split = split
        self.transform = transform

        split_values = [v for k,v in split.items()]
        assert np.round(sum(split_values)) == 1.0

        self.dataset_path = super().check_for_dataset()
        
        self.data_paths, self.runs = self.make_dataset(self)
        # self.time_init =     "Time:  0.00000E+00 y"
        self.time_first =    "   1 Time  1.00000E-01 y"
        self.time_final =    "   2 Time  5.00000E+00 y"
        # TODO put selection of input and output variables in a separate transform function (see ex4 - FeatureSelectorAndNormalizationTransform)
        self.input_vars_empty_value = PhysicalVariables(self.time_first, input_vars_names)
        self.output_vars_empty_value = PhysicalVariables(self.time_final, output_vars_names)
        self.datapoints = {}
        self.dimensions_of_datapoint:Tuple[int, int, int] = get_dimensions(f"{dataset_path}/{dataset_name}")

        
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
        # Split the data and runs into train, val and test
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
        
        if index not in self.datapoints.keys():
            self.datapoints[index] = self.load_datapoint(index)
            # print("created datapoint at index", index)

        return self.datapoints[index]

    def load_datapoint(self, index:int) -> DataPoint:
        """
        Load a datapoint at a given index (from data_paths) in the dataset with all input- and label-data
        and applies transforms if possible/ if any where given
        checks if the data is 2D or 3D - else: error
        """
        datapoint = DataPoint(index)

        datapoint.inputs = self._load_data_as_numpy(self.data_paths[index], self.input_vars_empty_value)
        datapoint.labels = self._load_data_as_numpy(self.data_paths[index], self.output_vars_empty_value)
        try:
            loc_hp_x = int(datapoint.get_loc_hp()[0])
            datapoint.inputs = self.transform(datapoint.inputs, loc_hp_x)
            datapoint.labels = self.transform(datapoint.labels, loc_hp_x)
        except Exception as e:
            print("no transforms applied: ", e)

        _assertion_error_2d(datapoint)
        return datapoint

    def reverse_transform(self, datapoint:DataPoint): #index:int, x_mean=None, x_std=None, y_mean=None, y_std=None):
        """
        Reverse the transformation of the data.
        """
        # data_dict = {}
        # index_material_id = self.get_input_properties().index('Material_ID')
        # data_dict["x"] = self.transform.reverse(self[index]["x"], mean=x_mean, std=x_std, index_material_id=index_material_id)
        # index_material_id = None
        # data_dict["y"] = self.transform.reverse(self[index]["y"], mean=y_mean, std=y_std)
        # data_dict["run_id"] = self.runs[index]

        datapoint.inputs = self.transform.reverse(datapoint.inputs)
        datapoint.labels = self.transform.reverse(datapoint.labels)

        _assertion_error_2d(datapoint)

        return datapoint

    def reverse_transform_temperature(self, temperature:Tensor, index_in_dataset:int) -> Tensor:
        mean_orig=self.datapoints[index_in_dataset].labels["Temperature [C]"].mean_orig
        std_orig=self.datapoints[index_in_dataset].labels["Temperature [C]"].std_orig
        temperature = self.transform.reverse_tensor_input(temperature, mean_orig=mean_orig, std_orig=std_orig)
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

        fraction_train = self.split['train']
        fraction_val = self.split['val']
        num_samples = len(data_paths)
        num_train = int(num_samples * fraction_train)
        num_valid = int(num_samples * fraction_val)
        
        np.random.seed(0)   # TODO remove later only for testing
        rand_perm = np.random.permutation(num_samples)
        
        if self.mode == 'train':
            indices = rand_perm[:num_train]
        elif self.mode == 'val':
            indices = rand_perm[num_train:num_train+num_valid]
        elif self.mode == 'test':
            indices = rand_perm[num_train+num_valid:]

        if isinstance(data_paths, list): 
            return list(np.array(data_paths)[indices]), list(np.array(labels)[indices])
        else: 
            return data_paths[indices], list(np.array(labels)[indices])
        
    def _load_data_as_numpy(self, data_path:str, variables:PhysicalVariables) -> PhysicalVariables:
        """
        Load data from h5 file on data_path, but only the variables named in variables.get_ids() at time stamp variables.time
        Sets the values of each PhysicalVariable in variables to the loaded data.
        """
        loaded_datapoint = PhysicalVariables(variables.time)
        # TODO when go to GPU: directly import as tensor?
        with h5py.File(data_path, "r") as file:
            for key, value in file[variables.time].items():
                if key in variables.get_ids_list(): # properties
                    loaded_datapoint[key] = np.array(value).reshape(self.dimensions_of_datapoint, order='F')
        return loaded_datapoint

'''NICHT ÜBERARBEITET
class MemoryImageFolderDataset(ImageFolderDataset):
    def __init__(self, root, *args,
                 transform=None,
                 download_url="https://i2dl.dvl.in.tum.de/downloads/cifar10memory.zip",
                 **kwargs):
        # Fix the root directory automatically
        if not root.endswith('memory'):
            root += 'memory'

        super().__init__(
            root, *args, download_url=download_url, **kwargs)
        
        with open(os.path.join(
            self.root_path, 'cifar10.pckl'
            ), 'rb') as f:
            save_dict = pickle.load(f)

        self.images = save_dict['images']
        self.labels = save_dict['labels']
        self.class_to_idx = save_dict['class_to_idx']
        self.classes = save_dict['classes']

        self.transform = transform

    def load_data_as_numpy(self, image_path):
        """Here we already have everything in memory,
        so we can just return the image"""
        return image_path
'''