# I am sorry for breaking the geneva convention on project structure
from pathlib import Path
from tqdm.auto import tqdm
import os
import torch

os.environ["KERAS_BACKEND"] = "torch"
import keras
import logging

import datetime
from torch.utils.tensorboard import SummaryWriter

import shutil
import matplotlib.pyplot as plt
import keras.ops as ops


def load_dataset(
    dataset_name="benchmark_dataset_2d_100datapoints_5years",
    dir="data",
    inputs_map=None,
    outputs_map=None,
):
    dataset_path = Path(dir) / dataset_name
    inputs_path = dataset_path / "Inputs"
    outputs_path = dataset_path / "Labels"
    available_inputs = list(os.listdir(inputs_path))
    all_inputs = []
    all_outputs = []
    for name in tqdm(available_inputs):
        try:
            new_input = torch.load(inputs_path / name)
            new_output = torch.load(outputs_path / name)
            if inputs_map is not None:
                new_input = inputs_map(new_input)
            if outputs_map is not None:
                new_output = outputs_map(new_output)
        except Exception as e:
            logging.warn(f"Could not load {name}: {e}")
        else:
            all_inputs.append(new_input)
            all_outputs.append(new_output)
    all_inputs = torch.stack(all_inputs)
    all_outputs = torch.stack(all_outputs)
    # we want channels last
    all_inputs = all_inputs.permute(0, 2, 3, 1)
    all_outputs = all_outputs.permute(0, 2, 3, 1)
    return ops.convert_to_tensor(all_inputs), ops.convert_to_tensor(all_outputs)


def coordinates(height, width, origin):
    """
    A grid of normalized coordinates, where the cell at `origin` is at (0,0)

    The normalization is such that the largest positive coordinate in each direction is normalized to 1

    Zoom also messes with the normalization.

    The result is of shape (height,width,2). The 2 is for (y, x)
    """
    y_range = ops.linspace(0, 1, height) - origin[0] / height
    x_range = ops.linspace(0, 1, width) - origin[1] / width
    X, Y = ops.meshgrid(x_range, y_range)
    return ops.stack((Y, X), axis=-1)


def dists_from_coords(coords):
    """
    Get a distortion field from a coordinate field

    Coords have shape (height,width,2)

    Output is shape (height,width,2,2), where (2,2) is the coordinate distortion vector (y,x) when moving up and right respectively

    The last row/column is not to be used in any transformation
    """
    h, w = coords.shape[:-1]
    dy = ops.diff(coords, axis=0)
    dx = ops.diff(coords, axis=1)
    dy = ops.concatenate((dy, dy[:1]))
    dx = ops.concatenate((dx, dx[:, :1]), axis=1)
    dist = ops.stack((dy, dx), axis=-1).reshape(h, w, 2, 2)
    return dist


class CustomTensorboard(keras.callbacks.Callback):
    def __init__(self, log_dir="logs", name = None):
        self.log_dir = Path(log_dir)
        if name is None:
            self.name = f"{datetime.datetime.now():%y-%m-%d %H-%M-%S}"
        else:
            self.name = name
        self.sw = SummaryWriter(self.log_dir / self.name)

    def on_epoch_end(self, epoch, logs=None):
        for metric, value in logs.items():
            self.sw.add_scalar(metric, value, epoch)


def almost_rectangle_sdf(coordinates, center, size):
    centered = coordinates - center
    distance = ops.abs(centered) - size / 2
    distance = ops.maximum(distance[..., 0], distance[..., 1])
    return ops.maximum(distance, 0)


class GlobalCoordinate(keras.layers.Layer):
    def __init__(self, coordinate_origin, oob_weight=1, **kwargs):
        # ortho loss broken at the moment
        super().__init__(**kwargs)
        self.coordinate_origin = ops.convert_to_tensor(
            coordinate_origin, dtype="float64"
        )
        self.offset = self.add_weight(
            shape=(2,),
            initializer=keras.initializers.Zeros(),
        )
        self.oob_weight = oob_weight

    def call(self, inputs):
        # input has shape (batch, height, width, channels), channels should be (2,2)
        batch, height, width, channels = inputs.shape
        tmp = ops.reshape(inputs, (*inputs.shape[:-1], 2, 2))
        oy, ox = 23, 7

        # (batch, height, to_the_right, dir=right, both vector elements)
        x_pos = ops.cumsum(tmp[..., ox:-1, 1, :], axis=-2)
        x_middle = ops.zeros((batch, height, 1, 2))
        # keras does not like negative stepsize in slices
        backwards_x = ops.flip(tmp[..., :, :ox, 1, :], axis=-2)
        # backwards_x = tmp[..., :, :ox, 1, :]
        x_neg = -ops.flip(ops.cumsum(backwards_x, axis=-2), axis=-2)

        y_pos = ops.cumsum(tmp[..., oy:-1, :, 0, :], axis=-3)
        y_middle = ops.zeros((batch, 1, width, 2))
        backwards_y = ops.flip(tmp[..., :oy, :, 0, :], axis=-3)
        # backwards_y = tmp[..., :oy, :, 0, :]
        y_neg = -ops.flip(ops.cumsum(backwards_y, axis=-3), axis=-3)
        from_dx = ops.concatenate((x_neg, x_middle, x_pos), axis=-2)
        from_dy = ops.concatenate((y_neg, y_middle, y_pos), axis=-3)
        coordinates = from_dx + from_dy + self.offset

        # tmp now has the change of coordinates from going in the x direction and the y direction in tmp[...,0,:] and tmp[...,1,:] respectively
        # tmp shape = (batch, height, width, 2,2)
        # from_dx = ops.cumsum(tmp[:, :-1, :, 0, :], axis=-3)
        # from_dx = ops.pad(from_dx, ((0, 0), (1, 0), (0, 0), (0, 0)))
        # from_dy = ops.cumsum(tmp[:, :, :-1, 1, :], axis=-2)
        # from_dy = ops.pad(from_dy, ((0, 0), (0, 0), (1, 0), (0, 0)))
        # coordinates = from_dx + from_dy - self.coordinate_origin + self.offset

        # TODO maybe scale this
        distances = almost_rectangle_sdf(
            coordinates, self.coordinate_origin, ops.array([2, 2])
        )
        oob_loss = ops.mean(distances) * self.oob_weight
        self.add_loss(oob_loss)

        return coordinates

    def compute_output_shape(self, input_shape):
        return *input_shape[:-1], 2

    def get_config(self):
        config = super().get_config()
        config.update({"coordinate_origin": self.coordinate_origin})
        return config


class Rotation(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = self.add_weight(shape=(), trainable=True, name="angle")

    def call(self, inputs):
        shape = inputs.shape
        inputs = inputs.reshape((-1, 2, 2))
        c = ops.cos(self.w)
        s = ops.sin(self.w)
        rot = ops.stack([c, -s, s, c]).reshape(2, 2)
        multiplied = inputs @ rot
        return multiplied.reshape(shape)

    def compute_output_shape(self, input_shape):
        return input_shape


class SaveOutputsCallback(keras.callbacks.Callback):
    def __init__(self, input, every=10):
        self.every = every
        self.input = input
        self.output_dir = Path("animation")


    def save_outputs(self, epoch):
        if epoch == 0:
            try:
                shutil.rmtree(self.output_dir)
            except Exception as e:
                logging.warning(f"Could not delete animation dir: {e}")
            self.output_dir.mkdir(exist_ok=True)
        result = self.model.predict(self.input, verbose=0)
        (oob_loss,) = [l.item() for l in self.model.losses]
        local_to_global = self.model.get_layer("Local to Global")
        nerf = self.model.get_layer("Pretrained Model").model
        to_global = local_to_global.get_layer("Global Coordinate")
        fig = plt.figure(figsize=(12, 10))
        plt.suptitle(
            f"epoch: {epoch}, offset = {to_global.offset.numpy()}\noob_loss: {oob_loss:.2e}"
        )
        size = (100,16)
        larger_coords = coordinates(*size, (23,7))
        nerf_output = nerf.predict(larger_coords.reshape(-1,2), batch_size= 10000, verbose = 0)
        nerf_output = nerf_output.reshape(size)
        plt.subplot(1,4,1)
        plt.title("Nerf output")
        plt.imshow(nerf_output)
        plt.colorbar()
        plt.subplot(1, 4, 2)
        plt.title("Temperature")
        plt.imshow(result[0, :, :, 0], vmin=0, vmax=1)
        plt.colorbar()
        plt.subplot(1, 4, 3)
        coords = local_to_global.predict(self.input, verbose=0)
        ys = coords[0, :, :, 0]
        xs = coords[0, :, :, 1]
        plt.title(f"y")
        plt.imshow(ys)
        plt.colorbar()
        plt.subplot(1, 4, 4)
        plt.title(f"x")
        plt.imshow(xs)
        plt.colorbar()
        plt.tight_layout()
        fig.savefig(f"animation/output_{epoch//self.every}.png")
        plt.close()

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.every == 0:
            self.save_outputs(epoch)

        return None


class ApplyToImage(keras.layers.Layer):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def call(self, inputs):
        shape = inputs.shape
        inputs = inputs.reshape((-1, 2))
        return self.model(inputs).reshape((*shape[:-1], 1))

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], 1)
