import os
import pathlib

import torch
import yaml
from torch import default_generator, randperm, Generator
from torch.utils.data import Dataset, Subset
from torch._utils import _accumulate
from typing import List,Optional,Sequence

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
        self.dp_per_run:int = ((self.spatial_size[0]) // self.box_size - 2) * (self.box_size // self.skip_per_dir) # -2 if exclude last box
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
        overlap = 46 # TODO HOTFIX for len_box=256, current UNetHalfPad2 architecture (3 blocks, 1 conv per block, 5x5 kernel, ...)
        assert self.box_size == 256, "HOTFIX for len_box=256, current UNetHalfPad2 architecture (3 blocks, 1 conv per block, 5x5 kernel, ...)"
        start_prior_box = box_id*self.skip_per_dir
        start_curr_box = start_prior_box + self.box_size -overlap
        
        input_curr = torch.load(self.path / "Inputs" / self.input_names[run_id])[:, start_curr_box : start_curr_box + self.box_size, :]
        input_prior_T = torch.load(self.path / "Labels" / self.input_names[run_id])[:, start_prior_box : start_prior_box + self.box_size, :]
        assert input_curr.shape[1:] == input_prior_T.shape[1:], f"Shapes of input and input_T do not match  {input_curr.shape}, {input_prior_T.shape}"
        input_all = torch.cat((input_curr, input_prior_T), dim=0)

        label = torch.load(self.path / "Labels" / self.label_names[run_id])[:, start_curr_box : start_curr_box+self.box_size, :]
        return input_all, label

    def idx_to_pos(self, idx):
        return idx // self.dp_per_run, idx % self.dp_per_run + 1 #TODO!!!! depends on which box is taken (front or last)
    
class DatasetEncoder(Dataset):
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
        self.dp_per_run:int = ((self.spatial_size[0]) // self.box_size - 2) * (self.box_size // self.skip_per_dir) # -2 if exclude last box
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
        label = torch.load(self.path / "Labels" / self.label_names[run_id])[:, box_id*self.skip_per_dir + self.box_size : box_id*self.skip_per_dir + self.box_size + 1, :]
        return input, label

    def idx_to_pos(self, idx):
        return idx // self.dp_per_run, idx % self.dp_per_run + 1

def get_splits(n, splits):
    splits = [int(n * s) for s in splits[:-1]]
    splits.append(n - sum(splits))
    return splits

def random_split_extend(dataset: DatasetExtend2, lengths: Sequence[int],
                 generator: Optional[Generator] = default_generator) -> List[Subset]:
    r"""
        Copy from torch.utils.data.dataset with adaptation to of 'blow up indices'
    """
    indices = randperm(sum(lengths), generator=generator).tolist()
    indices_extend = []
    for index in indices:
        indices_extend.extend([index*dataset.dp_per_run + i for i in range(dataset.dp_per_run)])

    lengths = [length * dataset.dp_per_run for length in lengths]

    return [Subset(dataset, indices_extend[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]