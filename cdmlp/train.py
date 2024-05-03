import os

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# TODO set cuda device *before* loading any keras modules
from cdmlp.models import CompleteModel
from cdmlp.util import load_and_split, manual_scheduler
import keras
from callbacks import CustomTensorboard, SaveOutputsCallback
from keras import ops
from pathlib import Path


def build_model(height, width):
    # This model gets x,y as input
    # as well as pressure gradient and permeability at the heatpumps location
    nerf = keras.Sequential(
        [
            keras.layers.Input(shape=(4,)),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(1, activation="leaky_relu"),
        ],
        name="NeRF-like",
    )

    dist_model = keras.Sequential(
        [
            keras.layers.Input(shape=(height, width, 4)),
            keras.layers.Conv2D(16, 3, activation="leaky_relu"),
            keras.layers.Conv2D(4, 3, activation="leaky_relu"),
        ],
        name="Distortion Model",
    )
    edge_size = 2

    complete_model = CompleteModel(
        nerf,
        dist_model,
        edge_size,
        ortho_weight=0,
    )
    return edge_size, complete_model


class RelativeLoss(keras.Loss):
    def __init__(self, eps=0.1):
        super().__init__(name="relative_loss")
        self.eps = eps

    def call(self, y_true, y_pred):
        return ops.mean(
            ops.abs(y_true - y_pred) / (ops.abs(y_true) + self.eps), axis=-1
        )


def train():
    run_name = "cdmlp new_vary_perm_data_vary_dist mae"
    (train_input, train_output), (val_input, val_output) = load_and_split(
        # "/scratch/sgs/pelzerja/datasets_prepared/1hp/dataset_small_10000dp_varyK_v3_part1 inputs_ksi plus_dummy_g",
        "/scratch/sgs/pelzerja/datasets_prepared/1hp/dataset_small_10000dp_varyK_v3_part2 inputs_gksi",
        # "/scratch/sgs/pelzerja_share/datasets_prepared/benchmark_dataset_2d_1000dp_vary_perm inputs_gksi",
        dir="",
        augment=True,
    )
    
    print(f"{train_output.device=}")
    print(f"{train_input['fields'].device=}")
    edge_size, complete_model = build_model(*(train_input["fields"]).shape[1:3],)
    complete_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.MeanAbsoluteError(),
        metrics=["mse"],
    )
    complete_model.fit(
        train_input,
        train_output[:, edge_size:-edge_size, edge_size:-edge_size, :],
        validation_data=(
            val_input,
            val_output[:, edge_size:-edge_size, edge_size:-edge_size, :],
        ),
        batch_size=32,
        epochs=10000,
        shuffle=True,
        callbacks=[
            CustomTensorboard("logs/complete_vary", name=run_name),
            # SaveOutputsCallback(inputs_train_vary[:1], outputs_train_vary[:1]),
            keras.callbacks.LearningRateScheduler(manual_scheduler(run_name)),
            keras.callbacks.ModelCheckpoint(
                (Path("checkpoints/complete_vary") / run_name).with_suffix(".keras")
            ),
            # keras.callbacks.BackupAndRestore("checkpoints/complete_vary/backup"),
        ],
        verbose=1,
    )
    save_path = Path("models") / run_name
    save_path.parent.mkdir(exist_ok=True, parents=True)
    complete_model.save(save_path.with_suffix(".keras"))


if __name__ == "__main__":
    train()
