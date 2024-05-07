import keras
from keras import ops
from cdmlp.util import coordinates, almost_rectangle_sdf
from cdmlp.layers import ValuesAroundPump, ApplyToImage
import math


class CompleteModel(keras.models.Model):
    def __init__(
        self,
        nerf,
        dist_model,
        edge_size,
        oob_weight=1,
        ortho_weight=0.1,
        mono_weight=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.edge_size = edge_size
        self.nerf = nerf
        self.dist_model = dist_model
        self.oob_weight = oob_weight
        self.ortho_weight = ortho_weight
        self.mono_weight = mono_weight
        self.local_to_global = GlobalCoordinate(
            dist_model,
            oob_weight=self.oob_weight,
            ortho_weight=self.ortho_weight,
            mono_weight=self.mono_weight,
            name="Coordinate Model",
        )
        self.apply_to_image = ApplyToImage(self.nerf, name="Nerf wrapper")
        self.values_around_pump = ValuesAroundPump(
            name="Values around pump"
        )

    def call(self, input_dict):
        global_coords = self.local_to_global.call(input_dict)
        s = self.edge_size
        fixed_values = self.values_around_pump.call(input_dict)[:, s:-s, s:-s, :]
        nerf_input = ops.concatenate((global_coords, fixed_values), axis=-1)
        return self.apply_to_image(nerf_input)

    def predict_coordinates(self, input_dict):
        return self.local_to_global.predict(input_dict, verbose=0)

    def predict_undisturbed(self, input_dict):
        # only works for single image
        batch, height, width, channels = input_dict["fields"].shape
        inputs = ops.empty((batch, height, width, 2))
        for index, (oy, ox) in enumerate(input_dict["pump_indices"]):
            inputs[index, ..., :2] = coordinates(height, width, (oy, ox))

        fixed_values = self.values_around_pump.__call__(input_dict)
        nerf_input = ops.concatenate((inputs, fixed_values), axis=-1)
        return self.apply_to_image.predict(nerf_input, verbose=0)

    def compute_output_shape(self, input_dict):
        return input_dict["fields"][:-1] + (1,)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "nerf": self.nerf,
                "dist_model": self.dist_model,
                "edge_size": self.edge_size,
                "oob_weight": self.oob_weight,
                "ortho_weight": self.ortho_weight,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["nerf"] = keras.saving.deserialize_keras_object(config["nerf"])
        config["dist_model"] = keras.saving.deserialize_keras_object(
            config["dist_model"]
        )
        return cls(**config)


class ApplyToImage(keras.models.Model):
    """
    Applies a scalar model (batch, channels_in) -> (batch, 1) to an image of shape (batch, height, width, channels)
    """

    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def call(self, inputs):
        shape = inputs.shape
        inputs = inputs.reshape((-1, shape[-1]))
        output_channels = self.model.output_shape[-1]
        return self.model(inputs).reshape((*shape[:-1], output_channels))

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


class GlobalCoordinate(keras.models.Model):
    def __init__(
        self,
        model,
        oob_weight=1,
        ortho_weight=0.1,
        mono_weight=1,
        **kwargs,
    ):
        # ortho loss broken at the moment
        super().__init__(**kwargs)
        self.model = model
        self.oob_weight = oob_weight
        self.ortho_weight = ortho_weight
        self.mono_weight = mono_weight

    def oob_loss(self, coordinates):
        distances = almost_rectangle_sdf(coordinates, ops.zeros(2), ops.array([2, 2]))
        return ops.mean(distances)

    def ortho_loss(self, deltas):
        eps = 1e-6
        max_angle = math.pi / 16
        dot_products = ops.sum(deltas[..., 0, :] * deltas[..., 1, :], axis=-1)
        norms = ops.sqrt(ops.sum(deltas * deltas, axis=-1))
        norm_product = norms[..., 0] * norms[..., 1] + eps
        return ops.mean(
            ops.maximum(
                ops.abs(dot_products / norm_product) - ops.cos(max_angle),
                0,
            )
        )

    def monotony_loss(self, deltas):
        # coords has shape (batch,height, width, 2,2)

        diff_y = ops.relu(-deltas[...,0,0])
        diff_x = ops.relu(-deltas[...,1,1])

        mean_y = ops.mean(diff_y)
        mean_x = ops.mean(diff_x)

        loss_y = mean_y / ops.mean(ops.abs(deltas[...,0,0]))
        loss_x = mean_x / ops.mean(ops.abs(deltas[...,1,1]))

        return loss_y + loss_x

    # def call(self, input_dict):
    #     # input has shape (batch, height, width, channels), channels should be (2,2)
    #     distortions = self.model(input_dict["fields"])
    #     batch, height, width, channels = distortions.shape
    #     deltas = ops.reshape(distortions, (*distortions.shape[:-1], 2, 2))
    #     coordinates = ops.zeros((batch, height, width, 2))
    #     for index, (oy, ox) in enumerate(input_dict["pump_indices"]):
    #         # keras does not like negative stepsize in slices
    #         backwards_x = ops.flip(deltas[index, :, :ox, 1, :], axis=-2)
    #         # backwards_x = tmp[..., :, :ox, 1, :]
    #         x_neg = -ops.flip(ops.cumsum(backwards_x, axis=-2), axis=-2)
    #         coordinates[index, :, :ox, :] = x_neg
    #         coordinates[index, :, ox + 1 :, :] = ops.cumsum(deltas[index, :, ox:-1, 1, :], axis=-2)

    #         backwards_y = ops.flip(deltas[index, :oy, :, 0, :], axis=-3)
    #         # backwards_y = tmp[..., :oy, :, 0, :]
    #         y_neg = -ops.flip(ops.cumsum(backwards_y, axis=-3), axis=-3)
    #         coordinates[index, :oy, :, :] += y_neg
    #         coordinates[index, oy + 1 :, :, :] += ops.cumsum(deltas[index, oy:-1, :, 0, :], axis=-3)

    #     self.add_loss(self.oob_loss(coordinates) * self.oob_weight)

    #     self.add_loss(self.ortho_loss(deltas) * self.ortho_weight)

    #     self.add_loss(self.monotony_loss(deltas) * self.mono_weight)

    #     return coordinates
    
    def call(self, input_dict):
    # input has shape (batch, height, width, channels), channels should be (2,2)
        distortions = self.model(input_dict["fields"])
        batch, height, width, channels = distortions.shape
        deltas = ops.reshape(distortions, (*distortions.shape[:-1], 2, 2))
        coordinates = ops.zeros((batch, height, width, 2))
        coordinates += ops.cumsum(deltas[..., 1, :], axis=-2)
        coordinates += ops.cumsum(deltas[..., 0, :], axis=-3)
        for index, (oy, ox) in enumerate(input_dict["pump_indices"]):
            coordinates[index, ...] -= coordinates[index, oy, ox, :].clone()

        self.add_loss(self.oob_loss(coordinates) * self.oob_weight)

        self.add_loss(self.ortho_loss(deltas) * self.ortho_weight)

        self.add_loss(self.monotony_loss(deltas) * self.mono_weight)

        return coordinates

    def compute_output_shape(self, input_shape):
        batch, height, width, channels = self.model.output_shape
        return batch, height, width, 2

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "model": self.model,
                "oob_weight": self.oob_weight,
                "ortho_weight": self.ortho_weight,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["model"] = keras.saving.deserialize_keras_object(config["model"])
        return cls(**config)
