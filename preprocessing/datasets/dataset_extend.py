import torch
from torch import Generator, default_generator, randperm
from torch._utils import _accumulate
from torch.utils.data import Subset
from typing import List, Optional, Sequence

from preprocessing.datasets.dataset import DatasetBasis

class DatasetExtend(DatasetBasis):
    def __init__(self, path:str, skip_per_dir:int=4, box_size:int=64):
        DatasetBasis.__init__(self, path, box_size)
        
        self.skip_per_dir:int = skip_per_dir
        self.dp_per_run:int = ((self.spatial_size[0]) // self.box_size - 2) * (self.box_size // self.skip_per_dir) # -2 if exclude last box
        print(f"dp_per_run: {self.dp_per_run}, spatial_size: {self.spatial_size}, box_size: {self.box_size}, skip_per_dir: {self.skip_per_dir}")

    @property
    def input_channels(self):
        return len(self.info["Inputs"])+1
    
    def __len__(self):
        return len(self.input_names) * self.dp_per_run

    def __getitem__(self, idx):
        run_id, box_id = self.idx_to_pos(idx)
        overlap = 46 # TODO HOTFIX
        start_first = 32
        assert self.box_size in [256, 128], "HOTFIX for current UNetHalfPad2 architecture (3 blocks, 1 conv per block, 5x5 kernel, ...)" # TODO only depends on architecture, not box_length??
        start_prior_box = box_id*self.skip_per_dir + start_first # to not learn from data around heat pump?
        start_curr_box = start_prior_box + self.box_size - overlap # 146
        
        input_curr = torch.load(self.path / "Inputs" / self.input_names[run_id])[:, start_curr_box : start_curr_box + self.box_size, :]
        input_prior_T = torch.load(self.path / "Labels" / self.input_names[run_id])[:, start_prior_box : start_prior_box + self.box_size, :]
        assert input_curr.shape[1:] == input_prior_T.shape[1:], f"Shapes of input and input_T do not match  {input_curr.shape}, {input_prior_T.shape}"
        input_all = torch.cat((input_curr, input_prior_T), dim=0)

        label = torch.load(self.path / "Labels" / self.label_names[run_id])[:, start_curr_box : start_curr_box+self.box_size, :]
        return input_all, label

    def idx_to_pos(self, idx):
        return idx // self.dp_per_run, idx % self.dp_per_run + 1 #TODO!!!! depends on which box is taken (front or last)
    
class DatasetEncoder(DatasetBasis):
    def __init__(self, path:str, skip_per_dir:int=4, box_size:int=64):
        DatasetBasis.__init__(self, path, box_size)
        self.skip_per_dir:int = skip_per_dir
        self.dp_per_run:int = ((self.spatial_size[0]) // self.box_size - 2) * (self.box_size // self.skip_per_dir) # -2 if exclude last box
        print(f"dp_per_run: {self.dp_per_run}, spatial_size: {self.spatial_size}, box_size: {self.box_size}, skip_per_dir: {self.skip_per_dir}")

    @property
    def input_channels(self):
        return len(self.info["Inputs"])+1
    
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
    
def random_split_extend(dataset: DatasetExtend, lengths: Sequence[int],
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