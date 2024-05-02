import keras
from keras import ops


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


class ApplyToImage(keras.layers.Layer):
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


class ValuesAroundPump(keras.layers.Layer):
    """
    Averages the values in channels around a pump.
    Example:
    input: (batch, height, width, channels)
    output: (batch, channels - 2)
    the last two channels are not used as they are assumed to be the coordinates
    """

    def __init__(self, radius=2, **kwargs):
        super().__init__(**kwargs)
        self.radius = radius

    def call(self, inputs):
        fields = inputs["fields"]
        batch, height, width, channels = fields.shape
        r = self.radius
        result = ops.zeros((batch, height, width, channels - 2))
        for index, (py, px) in enumerate(inputs["pump_indices"]):
            if self.radius is None:
                tmp = ops.mean(fields[index, ..., 2:], axis=(-2, -3), keepdims=True)
            else:
                tmp = ops.mean(
                    fields[index, py - r : py + r + 1, px - r : py + r + 1, 2:],
                    axis=(-2, -3),
                    keepdims=True,
                )
            result[index, :, :, :] = tmp

        return result

    def compute_output_shape(self, input_shape):
        batch, height, width, channels = input_shape
        return *input_shape[:-1], channels - 2

    def get_config(self):
        config = super().get_config()
        config.update({"radius": self.radius})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
