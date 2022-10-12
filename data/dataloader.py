"""Definition of Dataloader class"""

import numpy as np
from torch import Tensor, stack
import logging
from dataclasses import dataclass
from typing import Dict, List
from data.dataset import GWF_HP_Dataset

@dataclass
class DataLoader:
    """
    Defines an iterable batch-sampler over a given dataset
    """
    dataset:Dict[str, GWF_HP_Dataset]  #where to load the data from
    batch_size:int=1        #how many samples per batch to load
    shuffle:bool=False      #set to True to have the data reshuffled at every epoch
    drop_last:bool=False    #set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size. If False and the size of dataset is not divisible by the batch size, then the last batch will be smaller.

    def __iter__(self) -> List(Dict[str, Tensor]):
        """
        Returns the next batch of data with keywords like run_id, x, x_mean, y, ...; each referring to a
        Tensor with the shape of (batch_size, channels, H, W, (D))
        """
        def combine_batch_dicts(batch:List(Dict[str, GWF_HP_Dataset])) -> Dict[str, List(GWF_HP_Dataset)]:
            """
            Combines a given batch (list of dicts) to a dict of numpy arrays
            :param batch: batch, list of dicts
                e.g. [{k1: v1, k2: v2, ...}, {k1:, v3, k2: v4, ...}, ...]
            :returns: dict
                e.g. {k1: [v1, v3, ...], k2: [v2, v4, ...], ...}
            """
            batch_dict = {}
            for data_dict in batch:
                for key, value in data_dict.items():
                    if key not in batch_dict:
                        batch_dict[key] = []
                    batch_dict[key].append(value)
            return batch_dict

        def run_id_to_int(run_id:str) -> int:
            """
            :param run_id: "RUN_xx""
            :returns: int(xx)
            """
            return int(run_id.split("_")[-1])

        def batch_to_tensor(batch:Dict[str, List(GWF_HP_Dataset)]) -> Dict[str, Tensor]:
            """
            Returns a dict of tensors with keywords like run_id, x, x_mean, y, ...
            Tensor has the shape of (batch_size, C, H, W, (D))
            Transform all values of the given batch dict to tensors
            """
            tensor_batch = {}
            for key, values in batch.items():
                if key=="run_id": # expects the value not to be a Tensor and therefore needs to be converted to one
                    tensor_batch[key] = Tensor([run_id_to_int(id) for id in values]).float()
                else: # expects the value to be a Tensor  -> no convertion needed
                    try:
                        values = stack(value for value in values)
                    except:
                        # values contains only one element
                        logging.info("Values contains only one element?")
                        values = values[0]

                    if not isinstance(values, Tensor):
                        logging.info("careful: not a tensor so far - but don't worry, I'll take care of it")
                        tensor_batch[key] = Tensor(values)
                    else:
                        tensor_batch[key] = values
            return tensor_batch


        # shuffle and make iterator
        index_iterator = np.random.permutation(len(self.dataset)) if self.shuffle else range(len(self.dataset))
        index_iterator = iter(index_iterator)

        # get next batch
        batch = []
        for index in index_iterator:
            batch.append(self.dataset[index])
            if len(batch) == self.batch_size:
                yield batch_to_tensor(combine_batch_dicts(batch))
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yield batch_to_tensor(combine_batch_dicts(batch))
        

    def __len__(self) -> int:
        """
        Return the number of batches in the dataset, depending on whether drop_last is set
        """
        
        length = None
        if self.drop_last:
            length = len(self.dataset) // self.batch_size
        else:
            length = int(np.ceil(len(self.dataset) / self.batch_size))

        assert isinstance(length, int), "length is not an integer"
        return length