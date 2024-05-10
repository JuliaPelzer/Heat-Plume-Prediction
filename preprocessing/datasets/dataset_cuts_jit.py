import numpy as np
import torch

from preprocessing.datasets.dataset import DatasetBasis
from utils.utils_args import get_run_ids_from_prep

class SimulationDatasetCuts(DatasetBasis):
    def __init__(self, path:str, skip_per_dir:int=4, box_size:int=64, idx:int=0, case:str="train"):
        DatasetBasis.__init__(self, path, box_size)
        
        run_id = get_run_ids_from_prep(self.path / "Inputs")[idx]
        self.inputs = torch.load(self.path / "Inputs" / f"RUN_{run_id}.pt")
        self.labels = torch.load(self.path / "Labels" / f"RUN_{run_id}.pt")
        assert len(self.inputs.shape) == 3, "inputs should be 3D (C,H,W)"
        assert self.inputs.shape[1:] == self.labels.shape[1:], "inputs and labels have different shapes"
        self.spatial_size = self.inputs.shape[1:]
        assert self.spatial_size == self.labels.shape[1:], "inputs and labels have different spatial sizes" # TODO attention, if ever load several datapoints at once, this will fail
        self.box_size = np.array([box_size,box_size]) #512,512]) #[128,64] #[64, 32])
        self.box_out = np.array([0,0]).astype(int) #((self.box_size - np.array([244,244]))/2).astype(int)
        self.skip_per_dir = skip_per_dir
        self.case = case

    def __len__(self):
        return (self.spatial_size[0] - self.box_size[0]) * (self.spatial_size[1] - self.box_size[1]) // self.skip_per_dir**2
    
        # TODO only for split
        return int((self.spatial_size[0] - self.box_size[0]) * (self.spatial_size[1] - self.box_size[1]) // self.skip_per_dir**2 / 2) 

    def __getitem__(self, idx):
        pos = self.idx_to_pos(idx)
        # assert id too close to wall
        # assert (pos+self.box_size < self.spatial_size).all(), "box too close to wall" TODO too expensive if in every call?
        inputs = self.inputs[:, pos[0]:pos[0]+self.box_size[0], pos[1]:pos[1]+self.box_size[1]]
        labels = self.labels[:, pos[0]+self.box_out[0] : pos[0]+self.box_size[0]-self.box_out[0], pos[1]+self.box_out[1] : pos[1]+self.box_size[1]-self.box_out[1]]
        return inputs, labels

    def idx_to_pos(self, idx):
        return np.array([((idx*self.skip_per_dir) // ((self.spatial_size[1] - self.box_size[1])))*self.skip_per_dir, (idx*self.skip_per_dir) % ((self.spatial_size[1] - self.box_size[1]))])
    
        # TODO for split
        factor = 1 if self.case == "val" else 0
        idx_mod = int(((idx*self.skip_per_dir) // ((self.spatial_size[1] - self.box_size[1])))*self.skip_per_dir + factor*(self.spatial_size[0]-self.box_size[0])/2)
        idx_rest = (idx*self.skip_per_dir) % (self.spatial_size[1] - self.box_size[1])
        return np.array([idx_mod, idx_rest])