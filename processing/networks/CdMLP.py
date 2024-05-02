# set environment variable CUDA_VISIBLE_DEVICES to the GPU you want to use BEFORE loading keras
import os
os.environ["KERAS_BACKEND"] = "torch"

import torch
from pathlib import Path
import keras
from keras import ops
from cdmlp.util import coordinates
from cdmlp.models import CompleteModel  # This is needed for keras.models.load_model to work


class CdMLP:
    def __init__(self, model_path: Path):
        self.model = keras.models.load_model(model_path, custom_objects={"CompleteModel": CompleteModel})

    def apply(self, data_in: torch.Tensor, pump_indices: torch.Tensor, training_height=None, training_width=None,) -> torch.Tensor:
        """
        data_in: torch.Tensor of shape (batch_size, channels, height, width), e.g. channels = 4 (gksi)
        hp_locations: torch.Tensor of shape (batch_size, 2), with 2 being the y and x coordinates of the heat points (FIRST in height direction THEN width direction)
        data_out: torch.Tensor of shape (batch_size, channels, height, width), with channels = 1 (Temperature)
        training_width and training_height: int, the width and height of the training data
        """
        batch_size, channels, height, width = data_in.shape
        data = ops.empty((batch_size, height, width, 4))
        pump_indices = ops.convert_to_tensor(pump_indices)
        data[:, :, :, :2] = pump_indices
        for i, coord in enumerate(pump_indices):data[i, :, :, :2] = coordinates(height, width, coord, training_height, training_width)
        channels_last_data = ops.convert_to_tensor(data_in.permute(0, 2, 3, 1))
        data[:, :, :, 2:] = channels_last_data[..., :2]
        inputs = {"fields": data, "pump_indices": pump_indices}
        return self.model.predict(inputs, verbose=0)


if __name__ == "__main__":
    model_path = Path("models/pump_indices augmented.keras")
    model = CdMLP(model_path)
    data = torch.rand(1, 4, 64, 64)
    pump_indices = torch.tensor([[23, 9]])
    output = model.apply(data, pump_indices)
    print(output.shape)
