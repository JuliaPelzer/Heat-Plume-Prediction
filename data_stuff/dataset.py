import os
import pathlib

import torch
import yaml
from torch import default_generator, randperm, Generator
from torch.utils.data import Dataset, Subset
from torch._utils import _accumulate
from typing import List,Optional,Sequence
import numpy as np
import random

from data_stuff.transforms import NormalizeTransform
from processing.rotation import mask_tensor, rotate, get_rotation_angle, get_pressure_grad


class SimulationDataset(Dataset):
    def __init__(self, path):
        Dataset.__init__(self)
        self.path = pathlib.Path(path)
        self.input_names = []
        self.label_names = []
        for filename in os.listdir(self.path / "Inputs"):
            self.input_names.append(filename)
        for filename in os.listdir(self.path / "Labels"):
            self.label_names.append(filename)
        self.input_names.sort()
        self.label_names.sort()
        self.info = self.__load_info()
        self.norm = NormalizeTransform(self.info)

        if len(self.input_names) != len(self.label_names):
            raise ValueError(
                "Number of Inputs and labels does not match!")

    @property
    def input_channels(self):
        return len(self.info["Inputs"])

    @property
    def output_channels(self):
        return len(self.info["Labels"])

    def __load_info(self):
        with open(self.path.joinpath("info.yaml"), "r") as f:
            info = yaml.safe_load(f)
        return info

    def __len__(self):
        return len(self.input_names)

    def __getitem__(self, index):
        input = torch.load(self.path.joinpath(
            "Inputs", self.input_names[index]))
        label = torch.load(self.path.joinpath(
            "Labels", self.label_names[index]))
        return input, label
    
    def get_run_id(self, index):
        return self.input_names[index]

# class for build an augmented dataset 
class TrainDataset(Dataset):
    def __init__(self, path):
        Dataset.__init__(self)
        self.path = path
        self.info = self.__load_info()
        self.norm = NormalizeTransform(self.info)

        self.inputs = []
        self.labels = []
        self.run_ids = []

    @property
    def input_channels(self):
        return len(self.info["Inputs"])

    @property
    def output_channels(self):
        return len(self.info["Labels"])

    def __load_info(self):
        with open(self.path.joinpath("info.yaml"), "r") as f:
            info = yaml.safe_load(f)
        return info

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input = self.inputs[index]
        label = self.labels[index]
        return input, label
    
    def add_item(self, input, label, run_id):
        self.inputs.append(input)
        self.labels.append(label)
        self.run_ids.append(run_id)

    def get_run_id(self, index):
        return self.run_ids[index]
    
    @staticmethod
    def augment_data(dataset, augmentation_n : int = 0, mask : bool = False, angle : int = 0) -> Subset:
        """
        Augment data by adding rotated data points to the original dataset.

        Args:
            augmentation_n (int): Number of augmented data points to add. 
                                Setting to -1 does rotations of angles 90, 180, and 270 degrees.
            mask (bool): Apply circular mask to data fields if set to True.
            angle (int): Rotate all data points in original dataset by angle before augmenting (used for experimentation).
        
        Returns:
            torch.utils.data.Subset: Subset representing augmented dataset.
        """

        #get data from original dataset and apply circular mask/rotation
        if mask:
            inputs = [rotate(mask_tensor(dataset[i][0]),angle) for i in range(len(dataset))]
            labels = [rotate(mask_tensor(dataset[i][1]),angle) for i in range(len(dataset))]
        else:
            inputs = [rotate(dataset[i][0],angle) for i in range(len(dataset))]
            labels = [rotate(dataset[i][1],angle) for i in range(len(dataset))]

        run_ids = [dataset.dataset.get_run_id(i) for i in range(len(dataset))]
        
        augmented_dataset = TrainDataset(dataset.dataset.path)

        # add original data to output dataset
        for i in range(len(dataset)):
            augmented_dataset.add_item(inputs[i], labels[i], run_ids[i])
        
        # add augmented data points to output dataset 
        for i in range(len(dataset)):
            # add augmentation_n variations by uniformly sampling rotation angle from (0,360)
            for _ in range(augmentation_n):
                 rot_angle = np.random.rand()*360
                 augmented_dataset.add_item(rotate(inputs[i], rot_angle), rotate(labels[i], rot_angle), run_ids[i] + f'_rot_{rot_angle}')
            # extra augmentation mode used for 90*k degrees rotation
            if augmentation_n < 0:
                for rot_angle in [90,180,270]:
                    augmented_dataset.add_item(rotate(inputs[i], rot_angle), rotate(labels[i], rot_angle), run_ids[i] + f'_rot_{rot_angle}')

        return Subset(augmented_dataset, list(range(len(augmented_dataset))))
    
    # restrict dataset to data_n points, dont limit if data_n <= 0
    @staticmethod
    def restrict_data(dataset, data_n : int = -1, seed : int = 1) -> Subset:
        if data_n <= 0 or data_n >= len(dataset):
            return dataset
        
        random.seed(seed)
        
        restricted_dataset = TrainDataset(dataset.dataset.path) 
        data_points = random.sample([(dataset[i][0], dataset[i][1], dataset.dataset.get_run_id(i)) for i in range(len(dataset))], data_n)

        for data_point in data_points:
            restricted_dataset.add_item(data_point[0], data_point[1], data_point[2])
        print(f"LEN OF SUBSET IS {len(restricted_dataset)}")
        return Subset(restricted_dataset, list(range(len(restricted_dataset))))

    @staticmethod
    def rotate_data(dataset, grad_vec : list = [-1,0]):
        """
        Rotate data points in the dataset to align them with a specified direction.

        Args:
            dataset: Dataset containing data points to align.
            grad_vec (list): Direction vector to align data points to.
        
        Returns:
            TrainDataset: Dataset aligned to grad_vec direction.
        """
        #get data from original dataset
        inputs = [dataset[i][0] for i in range(len(dataset))]
        labels = [dataset[i][1] for i in range(len(dataset))]
        run_ids = [dataset.get_run_id(i) for i in range(len(dataset))]

        #normalize grad_vec
        grad_vec /= np.sqrt(grad_vec[0]**2 + grad_vec[1]**2) 
        
        rotated_dataset = TrainDataset(dataset.path)
        error = [0.,0.]
        # align dataset to grad_vec direction
        for i in range(len(dataset)):
            # align data point to grad_vec direction
            angle = get_rotation_angle(get_pressure_grad(inputs[i],dataset.info), grad_vec)
            rotated_input = rotate(inputs[i], angle)
            rotated_dataset.add_item(rotated_input, rotate(labels[i], angle), run_ids[i])
            
            # calculate error between the gradient of the rotated data point and the gradient that it has been aligned to
            rotated_grad = get_pressure_grad(rotated_input,dataset.info)
            # normalize gradient
            rotated_grad /= np.sqrt(rotated_grad[0]**2 + rotated_grad[1]**2) 
            error[0] += abs(grad_vec[0] - rotated_grad[0])
            error[1] += abs(grad_vec[1] - rotated_grad[1])
        error[0] /= len(rotated_dataset)
        error[1] /= len(rotated_dataset)
        print(f'Mean Average Error direction: {error}')

        return rotated_dataset

class DatasetExtend1(Dataset):
    def __init__(self, path:str, box_size:int=64):
        Dataset.__init__(self)
        self.path = pathlib.Path(path)
        self.info = self.__load_info()
        self.norm = NormalizeTransform(self.info)
        self.input_names = []
        self.label_names = []
        for filename in os.listdir(self.path / "Inputs"):
            self.input_names.append(filename)
        for filename in os.listdir(self.path / "Labels"):
            self.label_names.append(filename)
        self.input_names.sort()
        self.label_names.sort()
        self.spatial_size = torch.load(self.path / "Inputs" / self.input_names[0]).shape[1:]
        self.box_size = box_size

    @property
    def input_channels(self):
        return len(self.info["Inputs"])

    @property
    def output_channels(self):
        return len(self.info["Labels"])

    def __load_info(self):
        with open(self.path / "info.yaml", "r") as f:
            info = yaml.safe_load(f)
        return info

    def __len__(self):
        return len(self.input_names)
    
    def __getitem__(self, idx):
        input = torch.load(self.path / "Inputs" / self.input_names[idx])[:, :self.box_size, :]
        label = torch.load(self.path / "Labels" / self.label_names[idx])[:, :self.box_size, :]
        return input, label

class DatasetExtend2(Dataset):
    def __init__(self, path:str, skip_per_dir:int=4, box_size:int=64):
        Dataset.__init__(self)
        self.path = pathlib.Path(path)
        self.info = self.__load_info()
        self.norm = NormalizeTransform(self.info)
        self.input_names = []
        self.label_names = []
        for filename in os.listdir(self.path / "Inputs"):
            self.input_names.append(filename)
        for filename in os.listdir(self.path / "Labels"):
            self.label_names.append(filename)
        self.input_names.sort()
        self.label_names.sort()
        self.spatial_size = torch.load(self.path / "Inputs" / self.input_names[0]).shape[1:]
        self.box_size:int = box_size
        self.skip_per_dir:int = skip_per_dir
        self.dp_per_run:int = ((self.spatial_size[0]) // self.box_size - 2) * (self.box_size // self.skip_per_dir) #-2 to exclude last box
        print(f"dp_per_run: {self.dp_per_run}, spatial_size: {self.spatial_size}, box_size: {self.box_size}, skip_per_dir: {self.skip_per_dir}")

    @property
    def input_channels(self):
        return len(self.info["Inputs"])+1

    @property
    def output_channels(self):
        return len(self.info["Labels"])

    def __load_info(self):
        with open(self.path / "info.yaml", "r") as f:
            info = yaml.safe_load(f)
        return info
    
    def __len__(self):
        return len(self.input_names) * self.dp_per_run

    def __getitem__(self, idx):
        run_id, box_id = self.idx_to_pos(idx)
        input = torch.load(self.path / "Inputs" / self.input_names[run_id])[:, box_id*self.skip_per_dir + self.box_size : box_id*self.skip_per_dir + 2*self.box_size, :]
        input_T = torch.load(self.path / "Labels" / self.input_names[run_id])[:, box_id*self.skip_per_dir : box_id*self.skip_per_dir  + self.box_size, :]
        assert input.shape[1:] == input_T.shape[1:], f"Shapes of input and input_T do not match  {input.shape}, {input_T.shape}"
        input = torch.cat((input, input_T), dim=0)
        label = torch.load(self.path / "Labels" / self.label_names[run_id])[:, box_id*self.skip_per_dir + self.box_size : box_id*self.skip_per_dir + 2*self.box_size, :]
        return input, label

    def idx_to_pos(self, idx):
        return idx // self.dp_per_run, idx % self.dp_per_run + 1 #depends on which box is taken (front or last)
    
def get_splits(n, splits):
    splits = [int(n * s) for s in splits[:-1]]
    splits.append(n - sum(splits))
    return splits