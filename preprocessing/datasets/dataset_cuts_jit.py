import numpy as np
import torch
import pathlib

from preprocessing.datasets.dataset import DatasetBasis
from utils.utils_args import get_run_ids_from_prep

class SimulationDatasetCuts(DatasetBasis):
    def __init__(self, path:pathlib.Path, skip_per_dir:int=4, box_size:int=64, idx:int=0, case:str="train"):
        DatasetBasis.__init__(self, path, box_size)
        
        run_id = get_run_ids_from_prep(self.path / "Inputs")[idx]
        self.inputs = torch.load(self.path / "Inputs" / f"RUN_{run_id}.pt")
        self.labels = torch.load(self.path / "Labels" / f"RUN_{run_id}.pt")
        assert len(self.inputs.shape) == 3, "inputs should be 3D (C,H,W)"
        if not self.inputs.shape[1:] == self.labels.shape[1:]:
            required_shape = self.inputs.shape[1:]
            start_pos = [self.labels.shape[1]//2 - required_shape[0]//2, self.labels.shape[2]//2 - required_shape[1]//2]
            self.labels = self.labels[:, start_pos[0]:start_pos[0]+required_shape[0], start_pos[1]:start_pos[1]+required_shape[1]]
        self.spatial_size = self.inputs.shape[1:]
        self.box_size = np.array([box_size,box_size]) #512,512]) #[128,64] #[64, 32])
        self.box_out = np.array([0,0]).astype(int) #((self.box_size - np.array([244,244]))/2).astype(int)
        self.skip_per_dir = skip_per_dir
        self.case = case

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
        return np.array([((idx*self.skip_per_dir) // ((self.spatial_size[1] - self.box_size[1])))*self.skip_per_dir, (idx*self.skip_per_dir) % ((self.spatial_size[1] - self.box_size[1]))])