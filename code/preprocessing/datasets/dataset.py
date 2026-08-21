import os
import pathlib
import torch
import numpy as np
import yaml
from torch.utils.data import Dataset

from preprocessing.transforms import NormalizeTransform
from utils.utils_args import get_run_ids_from_prep

class DatasetBasis(Dataset):
    def __init__(self, path:str, box_size:int=None, cache: str = "none", cache_device: str = "cpu"):
        Dataset.__init__(self)
        self.path = pathlib.Path(path)
        self.info = self.__load_info()
        self.norm = NormalizeTransform(self.info)
        self.input_names = [filename for filename in os.listdir(self.path / "Inputs") if filename.endswith(".pt")]
        self.label_names = [filename for filename in os.listdir(self.path / "Labels") if filename.endswith(".pt")]
        self.input_names.sort()
        self.label_names.sort()
        self.__validate_matching_names()
        self.cache = {}
        self.cache_mode = cache.lower() if cache is not None else "none"
        self.cache_device = cache_device
        if self.cache_mode not in ["none", "cpu", "cuda"]:
            raise ValueError("cache must be one of: none, cpu, cuda")

        tmp_dp = torch.load(self.path / "Labels" / self.label_names[0], weights_only=False) # weights_only=False only for trusted sources
        self.n_output_channels = tmp_dp.shape[0]
        self.spatial_size = tmp_dp.shape[1:] # required for extend1,2 # TODO check if still works for extend (changed from Inputs to Labels for allin1)
        if box_size is not None:
            self.box_size = box_size
        else:
            self.box_size = self.spatial_size[0]

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

    def __validate_matching_names(self):
        input_names = set(self.input_names)
        label_names = set(self.label_names)
        if input_names != label_names:
            missing_labels = sorted(input_names - label_names)
            missing_inputs = sorted(label_names - input_names)
            raise ValueError(
                "Inputs and labels do not match by filename. "
                f"Missing labels for: {missing_labels}. "
                f"Missing inputs for: {missing_inputs}."
            )

    def _run_ids(self):
        return get_run_ids_from_prep(self.path / "Inputs")

    def __len__(self):
        return len(self.input_names)
    
    def __getitem__(self, i:int):
        if self.cache_mode == "none":
            input, label = self.__load_datapoint(i)
        else:
            if i not in self.cache:
                free_mem, total_mem = torch.cuda.mem_get_info()
                input, label = self.__load_datapoint(i)
                if self.cache_mode == "cuda": # and free_mem > 1024**4:
                    input = input.to(self.cache_device)
                    label = label.to(self.cache_device)
                self.cache[i] = input, label
            input, label = self.cache[i]
        return input, label

    def __load_datapoint(self, i:int):
        input = torch.load(self.path / "Inputs" / self.input_names[i], weights_only=False)[:, :self.box_size, :]
        label = torch.load(self.path / "Labels" / self.label_names[i], weights_only=False)[:, :self.box_size, :]
        return input, label

class DataPoint(DatasetBasis):
    def __init__(self, path:str, i:int=0, cache: str = "none", cache_device: str = "cpu"):
        DatasetBasis.__init__(self, path, cache=cache, cache_device=cache_device)
        if isinstance(i, int):
            run_id = self._run_ids()[i]
            
            self.input_names = [f"Sim_{run_id}.pt"]
            self.label_names = [f"Sim_{run_id}.pt"]
        elif isinstance(i, list) or isinstance(i, torch.Tensor) or isinstance(i, np.ndarray):
            indices = i.tolist() if hasattr(i, "tolist") else i
            run_ids = self._run_ids()
            self.input_names = [f"Sim_{run_ids[ii]}.pt" for ii in indices]
            self.label_names = [f"Sim_{run_ids[ii]}.pt" for ii in indices]
        else:
            raise ValueError("i must be an int or a list of ints")
        self.input_names.sort()
        self.label_names.sort()
