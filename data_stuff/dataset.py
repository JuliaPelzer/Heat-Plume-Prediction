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
from processing.rotation import mask_tensor, rotate, get_rotation_angle


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
    
    # augment data by adding rotated datapoints
    @staticmethod
    def augment_data(dataset, augmentation_n = 0, mask = False, angle = 0):
        if mask:
            inputs = [rotate(mask_tensor(dataset[i][0]),angle) for i in range(len(dataset))]
            labels = [rotate(mask_tensor(dataset[i][1]),angle) for i in range(len(dataset))]
        else:
            inputs = [rotate(dataset[i][0],angle) for i in range(len(dataset))]
            labels = [rotate(dataset[i][1],angle) for i in range(len(dataset))]

        run_ids = [dataset.dataset.get_run_id(i) for i in range(len(dataset))]
        
        augmented_dataset = TrainDataset(dataset.dataset.path)

        for i in range(len(dataset)):
            augmented_dataset.add_item(inputs[i], labels[i], run_ids[i])
        
        for i in range(len(dataset)):
            for _ in range(augmentation_n):
                 rot_angle = np.random.rand()*360
                 augmented_dataset.add_item(rotate(inputs[i], rot_angle), rotate(labels[i], rot_angle), run_ids[i] + f'_rot_{rot_angle}')
            #extra augmentation mode used for 90*k degrees rotation
            if augmentation_n < 0:
                for rot_angle in [90,180,270]:
                    augmented_dataset.add_item(rotate(inputs[i], rot_angle), rotate(labels[i], rot_angle), run_ids[i] + f'_rot_{rot_angle}')

        return Subset(augmented_dataset, list(range(len(augmented_dataset))))
    
    # restrict dataset to data_n points dont limit if data_n <= 0
    @staticmethod
    def restrict_data(dataset, data_n = -1, seed = 1):
        if data_n <= 0 or data_n >= len(dataset):
            return dataset
        
        random.seed(seed)
        
        restricted_dataset = TrainDataset(dataset.path) 
        data_points = random.sample([(dataset[i][0], dataset[i][1], dataset.get_run_id(i)) for i in range(len(dataset))], data_n)

        for data_point in data_points:
            restricted_dataset.add_item(data_point[0], data_point[1], data_point[2])

        return restricted_dataset
    
    # rotate datapoints such that pressure gradient points in grad_vec direction
    @staticmethod
    def rotate_data(dataset, grad_vec = [-1,0]):
        info = dataset.info
        p_ind = info['Inputs']['Liquid Pressure [Pa]']['index']
        
        center = int(dataset[0][0][p_ind].shape[0]/2)
        start = 5
        end = dataset[0][0][p_ind].shape[0] - 5
        dif = end - start

        inputs = [dataset[i][0] for i in range(len(dataset))]
        labels = [dataset[i][1] for i in range(len(dataset))]
        run_ids = [dataset.get_run_id(i) for i in range(len(dataset))]
        
        rotated_dataset = TrainDataset(dataset.path)

        for i in range(len(dataset)):
            angle = get_rotation_angle([(inputs[i][p_ind][end][center].item() - inputs[i][p_ind][start][center].item())/dif, 
                                    (inputs[i][p_ind][center][end].item() - inputs[i][p_ind][center][start].item())/dif], grad_vec)
            rotated_dataset.add_item(rotate(inputs[i], angle), rotate(labels[i], angle), run_ids[i])

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