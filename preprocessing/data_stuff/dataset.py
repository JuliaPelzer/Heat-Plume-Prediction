import os
import pathlib

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from preprocessing.data_stuff.transforms import NormalizeTransform

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
        # label = label[:, 512:1280, 512:1280]
        return input, label
    
    def get_run_id(self, index):
        return self.input_names[index]

class SimulationDatasetCuts(Dataset):
    def __init__(self, path:str, skip_per_dir:int=4):
        Dataset.__init__(self)
        self.path = pathlib.Path(path)
        self.info = self.__load_info()
        self.norm = NormalizeTransform(self.info)
        self.inputs = torch.load(self.path / "Inputs" / "RUN_0.pt")
        self.labels = torch.load(self.path / "Labels" / "RUN_0.pt")
        assert len(self.inputs.shape) == 3, "inputs should be 3D (C,H,W)"
        assert self.inputs.shape[1:] == self.labels.shape[1:], "inputs and labels have different shapes"
        self.spatial_size = self.inputs.shape[1:]
        assert self.spatial_size == self.labels.shape[1:], "inputs and labels have different spatial sizes" # TODO attention, if ever load several datapoints at once, this will fail
        self.box_size = np.array([64,64]) #512,512]) #[128,64] #[64, 32])
        self.box_out = np.array([0,0]).astype(int) #((self.box_size - np.array([244,244]))/2).astype(int)
        self.skip_per_dir = skip_per_dir

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
        return (self.spatial_size[0] - self.box_size[0]) * (self.spatial_size[1] - self.box_size[1]) // self.skip_per_dir**2

    def __getitem__(self, idx):
        pos = self.idx_to_pos(idx)
        # assert id too close to wall
        # assert (pos+self.box_size < self.spatial_size).all(), "box too close to wall" TODO too expensive if in every call?
        inputs = self.inputs[:, pos[0]:pos[0]+self.box_size[0], pos[1]:pos[1]+self.box_size[1]]
        labels = self.labels[:, pos[0]+self.box_out[0] : pos[0]+self.box_size[0]-self.box_out[0], pos[1]+self.box_out[1] : pos[1]+self.box_size[1]-self.box_out[1]]
        return inputs, labels

    def idx_to_pos(self, idx):
        # assert idx < (self.spatial_size[0] - self.box_size[0]) * (self.spatial_size[1] - self.box_size[1]) // self.skip_per_dir**2, "id out of range" # should later not be required because of __len__ TODO too expensive if in every call?
        return np.array([((idx*self.skip_per_dir) // ((self.spatial_size[1] - self.box_size[1])))*self.skip_per_dir, (idx*self.skip_per_dir) % ((self.spatial_size[1] - self.box_size[1]))])

def _get_splits(n, splits):
    splits = [int(n * s) for s in splits[:-1]]
    splits.append(n - sum(splits))
    return splits