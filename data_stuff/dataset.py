import os
import pathlib

import torch
import yaml
from torch import default_generator, randperm, Generator
from torch.utils.data import Dataset, Subset
#from torch._utils import _accumulate
from typing import List,Optional,Sequence
import matplotlib.pyplot as plt
import re
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
    
class DatasetExtendConvLSTM(Dataset):
    def __init__(self, path:str, prev_steps:int, extend:int, skip_per_dir:int, overfit:int):
        Dataset.__init__(self)
        self.path = pathlib.Path(path)
        self.info = self.__load_info()
        self.input_indices = [value['index'] for _, value in self.info['Inputs'].items()]
        self.norm = NormalizeTransform(self.info)
        self.input_names = []
        self.label_names = []
        self.inputs = []
        self.labels = []
        for filename in os.listdir(self.path / "Inputs"):
            self.input_names.append(filename)
            file_path = self.path / "Inputs" / filename
            self.inputs.append(torch.load(file_path))
            
        for filename in os.listdir(self.path / "Labels"):
            self.label_names.append(filename)
            file_path = self.path / "Labels" / filename
            self.labels.append(torch.load(file_path))

        #sort so that 'RUN_2' comes before 'RUN_10'
        self.input_names = sorted(self.input_names, key=self.natural_sort_key)
        self.label_names = sorted(self.label_names, key=self.natural_sort_key)

        self.spatial_size = torch.load(self.path / "Inputs" / self.input_names[0]).shape[1:]
        self.total_nr_steps = prev_steps + extend
        self.extend = extend
        self.box_len:int = (self.total_nr_steps) * self.spatial_size[1]
        self.skip_per_dir:int = skip_per_dir
        self.dp_per_run:int = (self.spatial_size[0] - self.box_len) // skip_per_dir
        self.overfit = overfit

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
    
    
    def __getitem__(self,idx):
        if self.overfit > 0:
            idx = idx % self.overfit
        run_id, window_nr = self.idx_to_window(idx)
        # file_path_inputs = self.path / "Inputs" / self.input_names[run_id]
        # file_path_labels = self.path / "Labels" / self.input_names[run_id]

        
        input = self.inputs[run_id]
        input = input[:, self.skip_per_dir*window_nr:self.skip_per_dir*window_nr+(self.box_len),:]

        temp = self.labels[run_id]
        temp = temp[:, self.skip_per_dir*window_nr:self.skip_per_dir*window_nr+(self.box_len),:] 

        input, temp = input.unsqueeze(1), temp.unsqueeze(1)
        input = torch.cat((input, temp), dim=0)
        input_slices = torch.tensor_split(input,self.total_nr_steps,axis=2) # produces array
        input_seq = torch.cat(input_slices, dim=1) # concatenates array elements to form a tensor
        temp_slices = torch.tensor_split(temp, self.total_nr_steps, axis=2)
        

        # set last temp-square to mean
        input_seq[-1,-1] = input_seq[-1].mean() #torch.zeros_like(input_seq[0,-64:])
        
        output = torch.cat(temp_slices, dim=1)[:,-self.extend:].reshape(1, self.extend * 64, 64)

        return input_seq, output

    def idx_to_window(self, idx):
        run_id = idx // self.dp_per_run
        window = idx % self.dp_per_run
        return run_id, window
    
    def natural_sort_key(self,s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]
    
def get_splits(n, splits):
    splits = [int(n * s) for s in splits[:-1]]
    splits.append(n - sum(splits))
    return splits