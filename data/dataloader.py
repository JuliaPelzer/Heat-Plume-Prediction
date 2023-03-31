"""Definition of Dataloader class"""

import logging
import numpy as np
from torch import Tensor, stack, cat, unsqueeze
from dataclasses import dataclass, field
from typing import Dict, List, Iterator


def _append_batch_by_datapoint(datapoint, batch) -> None:
    # after this: batch contains combined inputs, labels of all runs and channels of this batch
    # TODO do this earlier for more efficiency?
    assert batch.dim() == 1 or datapoint.dim() == batch.dim() - \
        1, "dimensions of batch and datapoint do not fit"

    if datapoint.dim() == batch.dim()-1:
        batch.inputs = cat(
            (batch.inputs, datapoint.inputs.unsqueeze(0)), dim=0)
        batch.labels = cat(
            (batch.labels, datapoint.labels.unsqueeze(0)), dim=0)
        # print("2st", datapoint.dim(), batch.dim(), Tensor.size(batch.inputs))
    elif batch.dim() == 1:
        dim_to_extend = 0
        batch.inputs = unsqueeze(datapoint.inputs, dim_to_extend)
        batch.labels = unsqueeze(datapoint.labels, dim_to_extend)
        # print("1nd", datapoint.dim(), batch.dim(), Tensor.size(batch.inputs))

    assert datapoint.dim() == batch.dim() - \
        1, "dimensions of batch do not fit to the datapoints"


def _append_tensor_in_0_dim(inputs) -> Tensor:
    dim_to_extend = 0
    first_in_var = list(inputs.keys())[0]
    assert isinstance(inputs[first_in_var].value,
                      Tensor), "inputs are not in tensor-format"

    result = inputs[first_in_var].value
    result = unsqueeze(result, dim_to_extend)
    for in_var in inputs.keys():
        if in_var != first_in_var:
            result = cat((result, inputs[in_var].value.unsqueeze(0)), dim=0)
    return result


def _run_id_to_int(run_id: str) -> int:
    """
    :param run_id: "RUN_xx""
    :returns: int(xx)
    """
    return int(run_id.split("_")[-1])
