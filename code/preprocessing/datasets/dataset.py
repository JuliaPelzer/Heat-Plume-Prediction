import enum
import os
import pathlib
from code.preprocessing.transforms import NormalizeTransform
from code.utils import logging as log  # noqa: F401
from code.utils.utils_args import get_run_ids_from_prep

import torch
import yaml
from torch.utils.data import Dataset


class DatasetType(enum.Enum):
    unknown = 0
    seasonal = 1
    steady_state_cooling = 2
    steady_state_heating = 3


class DatasetBasis(Dataset):
    def __init__(self, path: str, box_size: int = None, idx: int = None):
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
        self.spatial_size = torch.load(self.path / "Labels" / self.input_names[0]).shape[1:]
        if box_size is not None:
            self.box_size = box_size
        else:
            self.box_size = self.spatial_size[0]

        if len(self.input_names) != len(self.label_names):
            raise ValueError("Number of Inputs and labels does not match!")

    @property
    def input_channels(self):
        return len(self.info["Inputs"])

    @property
    def output_channels(self):
        return len(self.info["Labels"])

    def __load_info(self):
        with open(self.path / "info.yaml") as f:
            info = yaml.safe_load(f)
        return info

    def __len__(self):
        return len(self.input_names)

    def __getitem__(self, idx):
        input = torch.load(self.path / "Inputs" / self.input_names[idx])[:, : self.box_size, :]
        label = torch.load(self.path / "Labels" / self.label_names[idx])[:, : self.box_size, :]
        return input, label


class DataPoint(DatasetBasis):
    def __init__(self, path: str, i: int = 0):
        DatasetBasis.__init__(self, path)
        if isinstance(i, int):
            run_id = get_run_ids_from_prep(self.path / "Inputs")[i]

            self.input_names = [f"RUN_{run_id}.pt"]
            self.label_names = [f"RUN_{run_id}.pt"]
        elif isinstance(i, list):
            self.input_names = [f"RUN_{get_run_ids_from_prep(self.path / 'Inputs')[ii]}.pt" for ii in i]
            self.label_names = [f"RUN_{get_run_ids_from_prep(self.path / 'Labels')[ii]}.pt" for ii in i]
        else:
            raise ValueError("i must be an int or a list of ints")
        self.input_names.sort()
        self.label_names.sort()


class DataPointSequence(DataPoint):
    def __getitem__(self, idx):
        input = torch.load(self.path / "Inputs" / self.input_names[idx])[:, : self.box_size, :]
        label = torch.load(self.path / "Labels" / self.label_names[idx])[:, : self.box_size, :]
        inputs = torch.cat((input, label[:, 0, :, :].unsqueeze(1)), dim=0)
        return inputs, label
