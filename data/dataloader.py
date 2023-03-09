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
    dataset:DatasetSimulationData  #where to load the data from
    batch_size:int=10       #how many samples per batch to load
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
        batch_id = 0
        batch = Batch(batch_id=batch_id)
        for index in index_iterator:
            next_datapoint = _datapoint_to_tensor_including_channel(self.dataset[index])
            _append_batch_by_datapoint(next_datapoint, batch)
            if len(batch) == self.batch_size:
                yield batch.inputs, batch.labels
                batch_id += 1
                batch = Batch(batch_id=batch_id)
            
        if len(batch) > 0 and not self.drop_last:
            yield batch.inputs.float(), batch.labels.float()
    
    def reverse_transform(self, datapoint:Tensor=None):
        """
        Reverses the transformation of the datapoints, i.e. each datapoint is scaled back to the original
        physical values
        """
        if datapoint is not None:
            datapoint = self.dataset.reverse_transform(datapoint)
            return datapoint
        else:
            for data in self.dataset:
                data = self.dataset.reverse_transform(data)

    def reverse_transform_temperature(self, temperature:Tensor) -> Tensor:
        """
        Reverses the transformation of the output of a model - ! expects the output to be temperature
        """
        temperature = self.dataset.reverse_transform_temperature(temperature)
        return temperature

def _append_batch_by_datapoint(datapoint:Batch, batch:Batch) -> None:
    # after this: batch contains combined inputs, labels of all runs and channels of this batch
    # TODO do this earlier for more efficiency?
    assert batch.dim() == 1 or datapoint.dim() == batch.dim()-1, "dimensions of batch and datapoint do not fit"
    
    if datapoint.dim() == batch.dim()-1:
        batch.inputs = cat((batch.inputs, datapoint.inputs.unsqueeze(0)), dim=0)
        batch.labels = cat((batch.labels, datapoint.labels.unsqueeze(0)), dim=0)
        # print("2st", datapoint.dim(), batch.dim(), Tensor.size(batch.inputs))
    elif batch.dim() == 1:
        dim_to_extend = 0
        batch.inputs = unsqueeze(datapoint.inputs, dim_to_extend)
        batch.labels = unsqueeze(datapoint.labels, dim_to_extend)
        # print("1nd", datapoint.dim(), batch.dim(), Tensor.size(batch.inputs))

    assert datapoint.dim()==batch.dim()-1, "dimensions of batch do not fit to the datapoints"

def _append_tensor_in_0_dim(inputs:PhysicalVariables) -> Tensor:
    dim_to_extend = 0
    first_in_var = list(inputs.keys())[0]
    assert isinstance(inputs[first_in_var].value, Tensor), "inputs are not in tensor-format"

    result = inputs[first_in_var].value
    result = unsqueeze(result, dim_to_extend)
    for in_var in inputs.keys():
        if in_var != first_in_var:
            result = cat((result, inputs[in_var].value.unsqueeze(0)), dim=0)
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
