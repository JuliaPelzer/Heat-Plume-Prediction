import os
import pathlib

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from data_stuff.transforms import NormalizeTransform


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
        
        #TODO this shouldn't be done here

        x, _ = self.__getitem__(0)
        mat = x[0:2].unsqueeze(0)
        for i in range(1, self.__len__()):
            x, _ = self.__getitem__(i)
            mat = torch.cat((mat, x[0:2].unsqueeze(0)))
        num_sample, dim, n, m  = mat.shape
        mat_reshape = mat.reshape((num_sample, dim * n * m))
        mat_reshape = mat_reshape.swapaxes(0,1)
        self.U, self.S, self.Vh = torch.linalg.svd(mat_reshape, full_matrices=False)

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

def _get_splits(n, splits):
    splits = [int(n * s) for s in splits[:-1]]
    splits.append(n - sum(splits))
    return splits