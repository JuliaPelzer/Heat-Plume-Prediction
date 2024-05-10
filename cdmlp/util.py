from pathlib import Path
from tqdm.auto import tqdm
import torch
import logging
from keras import ops
from utils.utils_args import load_yaml

def load_dataset(
    dataset_name="benchmark_dataset_2d_100datapoints_5years",
    dir="data",
    inputs_map=None,
    outputs_map=None,
):
    dataset_path = Path(dir) / dataset_name
    inputs_path = dataset_path / "Inputs"
    outputs_path = dataset_path / "Labels"
    available_inputs = list(file.name for file in inputs_path.iterdir())
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


def load_and_split(dataset_name, dir="data", split=0.95, augment=False):
    info = load_yaml(Path(dir) / dataset_name / "info.yaml")
    pos_hp = info["PositionLastHP"][:2][::-1]
    raw_inputs, raw_outputs = load_dataset(
        dataset_name, dir=dir, inputs_map=lambda x: x[:2]
    )
    input_perm_log = ops.log(raw_inputs[..., info["Inputs"]["Permeability X [m^2]"]["index"]]+ops.exp(1))-1
    perm_log_min = ops.min(input_perm_log)
    perm_log_max = ops.max(input_perm_log)
    input_perm_log = (input_perm_log - perm_log_min) / (perm_log_max - perm_log_min)
    raw_inputs = ops.concatenate((raw_inputs, input_perm_log[..., None]), axis=-1)

    number, height, width, channels = raw_inputs.shape

    coordinate_origin_vary = ops.convert_to_tensor(pos_hp) # with orig dataset:[23,9]
    total_size = number
    if augment:
        total_size *= 4
    positions = ops.empty((total_size, 2), dtype=int)
    inputs = ops.empty((total_size, height, width, channels + 2))
    if not augment:
        coords = coordinates(height, width, coordinate_origin_vary)
        inputs[..., :2] = coords[None, ...]
        inputs[..., 2:] = raw_inputs

        outputs = raw_outputs
        positions[:] = coordinate_origin_vary[None, :]
    else:
        c = coordinate_origin_vary
        c_flipped_x = ops.convert_to_tensor([c[0], width - c[1] - 1])
        c_flipped_y = ops.convert_to_tensor([height - c[0] - 1, c[1]])
        c_flipped_both = ops.convert_to_tensor([height - c[0] - 1, width - c[1] - 1])
        coords_original = coordinates(height, width, c)
        coords_flipped_x = coordinates(height, width, c_flipped_x)
        coords_flipped_y = coordinates(height, width, c_flipped_y)
        coords_flipped_both = coordinates(height, width, c_flipped_both)
        positions[:number] = c[None, :]
        positions[number : 2 * number] = c_flipped_x[None, :]
        positions[2 * number : 3 * number] = c_flipped_y[None, :]
        positions[3 * number :] = c_flipped_both[None, :]
        inputs[:number, ..., :2] = coords_original[None, ...]
        inputs[number : 2 * number, ..., :2] = coords_flipped_x[None, ...]
        inputs[2 * number : 3 * number, ..., :2] = coords_flipped_y[None, ...]
        inputs[3 * number :, ..., :2] = coords_flipped_both[None, ...]
        inputs[:number, ..., 2:] = raw_inputs
        inputs[number : 2 * number, ..., 2:] = ops.flip(raw_inputs, axis=-2)  # flip x
        inputs[2 * number : 3 * number, ..., 2:] = ops.flip(
            raw_inputs, axis=-3
        )  # flip y
        inputs[3 * number :, ..., 2:] = ops.flip(raw_inputs, axis=(-2, -3))  # flip both
        # now hackily invert the pressure gradient for the y flips
        inputs[2 * number :, ..., 2] *= -1

        os = raw_outputs.shape
        outputs = ops.empty((number * 4, *os[1:]))
        outputs[:number] = raw_outputs
        outputs[number : 2 * number] = ops.flip(raw_outputs, axis=-2)  # flip x
        outputs[2 * number : 3 * number] = ops.flip(raw_outputs, axis=-3)  # flip y
        outputs[3 * number :] = ops.flip(raw_outputs, axis=(-2, -3))  # flip both

    if split is None:
        return {"fields": inputs, "pump_indices": positions}, outputs, info
    split = 0.95
    n_split = int(split * total_size)
    inputs_train, inputs_val = inputs[:n_split], inputs[n_split:]
    outputs_train, outputs_val = (
        outputs[:n_split],
        outputs[n_split:],
    )
    positions_train = positions[:n_split]
    positions_val = positions[n_split:]
    # now inputs is: y,x, pressure gradient, permeability
    logging.debug(f"{inputs.shape=}")
    logging.debug(f"{outputs.shape=}")
    logging.debug(f"{inputs.device=}")
    return ({"fields": inputs_train, "pump_indices": positions_train},outputs_train,), ({"fields": inputs_val, "pump_indices": positions_val}, outputs_val), info


def coordinates(
    height,
    width,
    origin,
    training_height=None,
    training_width=None,
    super_resolution=1,
):
    """
    A grid of normalized coordinates, where the cell at `origin` is at (0,0)

    The normalization is such that the largest positive coordinate in each direction is normalized to 1

    Zoom also messes with the normalization.

    The result is of shape (height,width,2). The 2 is for (y, x)
    """
    if training_height is None:
        training_height = height
    if training_width is None:
        training_width = width
    y_range = (
        ops.linspace(
            0, 1 * height / training_height, height * super_resolution, endpoint=False
        )
        - origin[0] / training_height
    )
    x_range = (
        ops.linspace(
            0, 1 * width / training_width, width * super_resolution, endpoint=False
        )
        - origin[1] / training_width
    )
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


def almost_rectangle_sdf(coordinates, center, size):
    centered = coordinates - center
    distance = ops.abs(centered) - size / 2
    distance = ops.maximum(distance[..., 0], distance[..., 1])
    return ops.maximum(distance, 0)


def manual_scheduler(run_name="default", dir="learning_rates"):
    dir = Path(dir)
    file = dir / f"{run_name}.txt"
    file.parent.mkdir(exist_ok=True, parents=True)

    def wrapper(epoch, lr):
        if epoch == 0:
            file.write_text(f"{lr:.2e}\n")
            return lr
        else:
            try:
                text = file.read_text()
                lr = float(text)
            except:
                pass
            return lr

    return wrapper
