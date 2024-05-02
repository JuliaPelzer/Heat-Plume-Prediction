import keras
import datetime
from tensorboardX import SummaryWriter
from pathlib import Path
import shutil
import logging
import matplotlib.pyplot as plt


class CustomTensorboard(keras.callbacks.Callback):
    def __init__(self, log_dir="logs", name=None):
        self.log_dir = Path(log_dir)
        if name is None:
            self.name = f"{datetime.datetime.now():%y-%m-%d %H-%M-%S}"
        else:
            self.name = name
        self.sw = SummaryWriter(self.log_dir / self.name, flush_secs=1)

    def on_epoch_end(self, epoch, logs=None):
        for metric, value in logs.items():
            self.sw.add_scalar(metric, value, epoch)
        self.sw.add_scalar(
            "learning_rate", self.model.optimizer.get_config()["learning_rate"], epoch
        )


class SaveOutputsCallback(keras.callbacks.Callback):
    """
    Saves an image of the nerf output, the temperature, and the coordinates every `every` epochs.
    The output dir is automatically cleared on the 0th epoch
    """

    def __init__(self, input, label=None, output_dir="animation", every=10):
        self.every = every
        self.input = input
        self.output_dir = Path(output_dir)
        self.label = label

    def save_outputs(self, epoch):
        if epoch == 0:
            try:
                shutil.rmtree(self.output_dir)
            except Exception as e:
                logging.warning(f"Could not delete animation dir: {e}")
            self.output_dir.mkdir(exist_ok=True)
        result = self.model.predict(self.input, verbose=0)
        (oob_loss, ortho_loss) = [l.item() for l in self.model.losses]
        fig = plt.figure(figsize=(12, 10))
        plt.suptitle(
            f"epoch: {epoch}, \noob_loss: {oob_loss:.2e}, ortho_loss: {ortho_loss:.2e}"
        )
        nerf_output = keras.ops.convert_to_numpy(
            self.model.predict_undisturbed(self.input)[0]
        )
        plt.subplot(1, 5, 1)
        plt.title("Nerf output")
        plt.imshow(nerf_output, vmin=0, vmax=1)
        plt.colorbar()
        plt.subplot(1, 5, 2)
        plt.title("Temperature")
        plt.imshow(result[0, :, :, 0], vmin=0, vmax=1)
        plt.colorbar()
        plt.subplot(1, 5, 3)
        plt.title("target")
        if self.label is not None:
            plt.imshow(
                keras.ops.convert_to_numpy(self.label[0, :, :, 0]), vmin=0, vmax=1
            )
            plt.colorbar()
        coords = self.model.predict_coordinates(self.input)
        ys = coords[0, :, :, 0]
        xs = coords[0, :, :, 1]
        plt.subplot(1, 5, 4)
        plt.title(f"x")
        plt.imshow(xs)
        plt.colorbar()
        plt.subplot(1, 5, 5)
        plt.title(f"y")
        plt.imshow(ys)
        plt.colorbar()
        plt.tight_layout()
        fig.savefig(f"animation/output_{epoch//self.every}.png")
        plt.close()

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.every == 0:
            self.save_outputs(epoch)

        return None


class TrainEpoch(keras.callbacks.Callback):
    def __init__(self, fit_model, **fit_kwargs):
        self.fit_model = fit_model
        self.fit_kwargs = fit_kwargs

    def on_epoch_end(self, epoch, logs=None):
        self.fit_model.fit(**self.fit_kwargs)
