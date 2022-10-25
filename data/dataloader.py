"""Definition of Dataloader class"""

import logging
import numpy as np
from torch import Tensor, stack, cat, unsqueeze
from dataclasses import dataclass, field
from typing import Dict, List, Iterator
from data.dataset import DatasetSimulationData
from data.utils import Batch, DataPoint, PhysicalVariables

@dataclass
class DataLoader:
    """
    Defines an iterable batch-sampler over a given dataset
    """
    dataset:Dict[str, DatasetSimulationData]  #where to load the data from
    batch_size:int=1        #how many samples per batch to load
    shuffle:bool=False      #set to True to have the data reshuffled at every epoch
    drop_last:bool=False    #set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size. If False and the size of dataset is not divisible by the batch size, then the last batch will be smaller.

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

    def __iter__(self) -> Iterator[List[Dict[str, Tensor]]]:
        """
        Returns the next batch of data with keywords like run_id, x, x_mean, y, ...; each referring to a
        Tensor with the shape of (batch_size, channels, H, W, (D))
        TODO Comment
        """
        # shuffle and make iterator
        index_iterator = np.random.permutation(len(self.dataset)) if self.shuffle else range(len(self.dataset))
        index_iterator = iter(index_iterator)

        # get next batch
        batch = Batch(batch_id=0)
        for index in index_iterator:
            next_datapoint = _datapoint_to_tensor_including_channel(self.dataset[index])


        # get next batch
        batch:List[DataPoint] = []
        for index in index_iterator:
            print(self.dataset[index])
            batch.append(self.dataset[index])
            if len(batch) == self.batch_size:
                yield _batch_to_tensor(_combine_batch_dicts(batch))
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yield _batch_to_tensor(_combine_batch_dicts(batch))

def _batch_to_tensor(batch:Dict[str, List[DatasetSimulationData]]) -> Dict[str, Tensor]:
    """
    Returns a dict of tensors with keywords like run_id, x, x_mean, y, ...
    Tensor has the shape of (batch_size, C, H, W, (D))
    Transform all values of the given batch dict to tensors
    """
    tensor_batch = {}
    for key, values in batch.items():
        if key=="run_id": # expects the value not to be a Tensor and therefore needs to be converted to one
            tensor_batch[key] = Tensor([_run_id_to_int(id) for id in values]).float()
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

def _combine_batch_dicts(batch:List[Dict[str, DatasetSimulationData]]) -> Dict[str, List[DatasetSimulationData]]:
    """
    Combines a given batch (list of dicts) to a dict of lists
    :param batch: batch, list of dicts
        e.g. [{k1: v1, k2: v2, ...}, {k1:, v3, k2: v4, ...}, ...]
    :returns: dict
        e.g. {k1: [v1, v3, ...], k2: [v2, v4, ...], ...}
    """
    # input: List[DataPoint]
    # output: Dict[str, List[DataPoint]]
    # batch_dict = {"inputs": List[values], "labels": List[values]}"}
    batch_dict = {}
    for data_dict in batch:
        for key, value in data_dict.items():
            if key not in batch_dict:
                batch_dict[key] = []
            batch_dict[key].append(value)
    return batch_dict

def _combine_datapoints_to_batch(datapoints: Batch) -> Batch:
    pass

def _append_tensor_in_0_dim(inputs:PhysicalVariables) -> Tensor:
    dim_to_extend = 0
    first_in_var = list(inputs.keys())[0]
    result = inputs[first_in_var].value
    result = unsqueeze(result, dim_to_extend)
    print("result init", result.shape)
    for in_var in inputs.keys():
        print("inputs", inputs[in_var].value.shape)
        if in_var != first_in_var:
            result = cat((result, inputs[in_var].value.unsqueeze(0)), dim=0)
    print("result", result.shape)
    return result

def _datapoint_to_tensor_including_channel(datapoint:DataPoint) -> Batch:
    combined_values = Batch(datapoint.run_id) # initialize batch
    combined_values.inputs = _append_tensor_in_0_dim(datapoint.inputs)
    combined_values.labels = _append_tensor_in_0_dim(datapoint.labels)
    return combined_values
    

def _run_id_to_int(run_id:str) -> int:
    """
    :param run_id: "RUN_xx""
    :returns: int(xx)
    """
    return int(run_id.split("_")[-1])
