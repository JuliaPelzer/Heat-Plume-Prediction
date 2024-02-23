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
    def __init__(self, pump_indices, oob_weight=1, **kwargs):
        # ortho loss broken at the moment
        super().__init__(**kwargs)
        self.pump_indices = ops.convert_to_tensor(pump_indices)
        self.oob_weight = oob_weight
    
    def build(self, input_shape):
        self.offset = self.add_weight(
            shape=(2,),
            initializer=keras.initializers.Zeros(),
            name = "offset"
        )

    def call(self, inputs):
        # input has shape (batch, height, width, channels), channels should be (2,2)
        batch, height, width, channels = inputs.shape
        tmp = ops.reshape(inputs, (*inputs.shape[:-1], 2, 2))
        oy, ox = self.pump_indices

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

        # TODO maybe scale this
        distances = almost_rectangle_sdf(
            coordinates, ops.zeros(2), ops.array([2, 2])
        )
        oob_loss = ops.mean(distances) * self.oob_weight
        self.add_loss(oob_loss)

        return coordinates

    def compute_output_shape(self, input_shape):
        return *input_shape[:-1], 2

    def get_config(self):
        config = super().get_config()
        config.update({"pump_indices": self.pump_indices, "oob_weight": self.oob_weight})
        return config
    
    @classmethod
    def from_config(cls, config):
        config["pump_indices"] = keras.saving.deserialize_keras_object(config["pump_indices"])
        return cls(**config)


class Rotation(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def build(self, input_shape):
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
        fig = plt.figure(figsize=(12, 10))
        plt.suptitle(
            f"epoch: {epoch}, \noob_loss: {oob_loss:.2e}"
        )
        size = (100,16)
        nerf_output = self.model.predict_undisturbed(self.input)[0]
        plt.subplot(1,4,1)
        plt.title("Nerf output")
        plt.imshow(nerf_output)
        plt.colorbar()
        plt.subplot(1, 4, 2)
        plt.title("Temperature")
        plt.imshow(result[0, :, :, 0], vmin=0, vmax=1)
        plt.colorbar()
        plt.subplot(1, 4, 3)
        coords = self.model.predict_coordinates(self.input)
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
        inputs = inputs.reshape((-1, shape[-1]))
        return self.model(inputs).reshape((*shape[:-1], 1))

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], 1)
    
    def get_config(self):
        config = super().get_config()
        config.update({"model": self.model})
        return config
    
    @classmethod
    def from_config(cls, config):
        config["model"] = keras.saving.deserialize_keras_object(config["model"])
        return cls(**config)

class ValuesAroundPump(keras.layers.Layer):
    def __init__(self, pump_indices, **kwargs):
        super().__init__(**kwargs)
        self.pump_indices = ops.convert_to_tensor(pump_indices)

    def call(self, inputs):
        batch,height,width,channels = inputs.shape
        py,px = self.pump_indices
        all_values = inputs[:,py,px, :-2]
        # last two are coordinates, so we don't want them
        return ops.reshape(ops.repeat(all_values[:,None,:], height*width, axis = 1), (batch, height, width, -1))

    def compute_output_shape(self, input_shape):
        batch,height,width,channels = input_shape
        return *input_shape[:-1], channels - 2
    
    def get_config(self):
        config = super().get_config()
        config.update({"pump_indices": self.pump_indices})
        return config
    
    @classmethod
    def from_config(cls, config):
        config["pump_indices"] = keras.saving.deserialize_keras_object(config["pump_indices"])
        return cls(**config)
    
class CompleteModel(keras.models.Model):
    def __init__(self,pump_indices, nerf, dist_model,edge_size, **kwargs):
        super().__init__(**kwargs)
        self.edge_size = edge_size
        self.pump_indices = pump_indices
        self.nerf = nerf
        self.dist_model = dist_model
        self.local_to_global = keras.Sequential(
            [
                keras.layers.Input(shape=dist_model.input_shape[1:]),
                dist_model,
                GlobalCoordinate(self.pump_indices,oob_weight = 1, name = "Global Coordinate"),
            ],
            name="Local to Global",
        )
        self.apply_to_image = ApplyToImage(self.nerf, name="Nerf wrapper")
        self.apply_to_image_model = keras.Sequential(
            [
                keras.layers.Input(shape=(*dist_model.input_shape[1:-1],4)),
                self.apply_to_image
            ]
        )
        self.values_around_pump = ValuesAroundPump(self.pump_indices, name = "Values around pump")

    def call(self, inputs):
        global_coords = self.local_to_global(inputs)
        s = self.edge_size
        fixed_values = self.values_around_pump(inputs)[:,s:-s,s:-s,:]
        nerf_input = ops.concatenate((global_coords, fixed_values), axis = -1)
        return self.apply_to_image(nerf_input)
    
    def predict_coordinates(self,inputs):
        return self.local_to_global.predict(inputs, verbose = 0)
    
    def predict_undisturbed(self, inputs):
        # only works for single image
        batch, height, width, channels = inputs.shape
        coords = coordinates(height,width, self.pump_indices).reshape(1,height,width,-1)
        fixed_values = self.values_around_pump.__call__(inputs)
        nerf_input = ops.concatenate((coords, fixed_values), axis = -1)
        return self.apply_to_image_model.predict(nerf_input, verbose = 0)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (1,)
    
    def get_config(self):
        config = super().get_config()
        config.update({"pump_indices": self.pump_indices, "nerf": self.nerf, "dist_model": self.dist_model, "edge_size": self.edge_size})
        return config
    
    @classmethod
    def from_config(cls, config):
        config["pump_indices"] = keras.saving.deserialize_keras_object(config["pump_indices"])
        config["nerf"] = keras.saving.deserialize_keras_object(config["nerf"])
        config["dist_model"] = keras.saving.deserialize_keras_object(config["dist_model"])
        return cls(**config)