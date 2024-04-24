import os
import pathlib
import torch
import yaml
from torch.utils.data import Dataset

from preprocessing.transforms import NormalizeTransform
from utils.utils_args import get_run_ids_from_prep

class DatasetBasis(Dataset):
    def __init__(self, path:str, box_size:int=None, idx:int=None):
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
        self.spatial_size = torch.load(self.path / "Inputs" / self.input_names[0]).shape[1:] # required for extend1,2
        if box_size is not None:
            self.box_size = box_size # required for extend1,2
        else:
            self.box_size = self.spatial_size[0]

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
        with open(self.path / "info.yaml", "r") as f:
            info = yaml.safe_load(f)
        return info

    def __len__(self):
        return len(self.input_names)
    
    def __getitem__(self, idx):
        input = torch.load(self.path / "Inputs" / self.input_names[idx])[:, :self.box_size, :]
        label = torch.load(self.path / "Labels" / self.label_names[idx])[:, :self.box_size, :]
        return input, label

class DataPoint(DatasetBasis):
    def __init__(self, path:str, idx:int=0):
        DatasetBasis.__init__(self, path)
        run_id = get_run_ids_from_prep(self.path / "Inputs")[idx]
        
        self.input_names = [f"RUN_{run_id}.pt"]
        self.label_names = [f"RUN_{run_id}.pt"]
        
def get_splits(n, splits):
    splits = [int(n * s) for s in splits[:-1]]
    splits.append(n - sum(splits))
    return splits