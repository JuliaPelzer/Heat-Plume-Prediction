from code.postprocessing.visualization import visualize_streamlines
from code.preprocessing.datasets.dataset import DatasetType
from code.preprocessing.preparing_datasets.raw_data_loading import get_hp_location_raw_np
from code.preprocessing.preprocessing import expand_property_names
from code.preprocessing.transforms import NormalizeTransform
from code.processing.networks.unetVariants import UNetNoPad2
from code.utils import logging as log  # noqa: F401
from pathlib import Path

import numpy as np
import torch


def load_velocity_field(
    run_id: str,
    origin_path: Path,
    use_model: bool,
    model: UNetNoPad2 | None,
    norm_transform: NormalizeTransform,
    device: torch.device,
) -> torch.Tensor:
    """Loads velocity from disk or infers it using the UNet model."""
    if use_model and model is not None:
        data_in = torch.load(origin_path / "Inputs" / run_id)
        with torch.no_grad():
            velocity = model(data_in.unsqueeze(0).to(device)).cpu().detach().squeeze(0)
    else:
        velocity = torch.load(origin_path / "Labels" / run_id)

    norm_transform.reverse(velocity, "Labels")
    return velocity


def extract_heat_pump_positions(input_tensor: torch.Tensor, channel_index: int) -> np.ndarray:
    """Finds coordinates where Material ID == 1 (Heat Pumps)."""
    # Ensure tensor is on CPU before converting to numpy
    pos_hps = get_hp_location_raw_np(input_tensor[channel_index].detach().cpu().numpy()).T.astype(float)
    return pos_hps + 0.5


def save_result(
    destination_path: Path,
    run_id: str,
    inputs_tensor: torch.Tensor,
    streamlines: dict[str, torch.Tensor],
    norm_transform: NormalizeTransform,
    step3_in_map: dict[str, int],
    step2_in_map: dict[str, int],
) -> None:
    """Renormalizes and saves the final tensor to disk."""
    inputs_normed = norm_transform(inputs_tensor, "Inputs")

    # Inject calculated streamlines for keys present in step3 but missing in step2
    for key, idx in step3_in_map.items():
        if key not in step2_in_map and key in streamlines:
            inputs_normed[idx] = streamlines[key]

    output_dir = destination_path / "Inputs"
    output_dir.mkdir(exist_ok=True, parents=True)

    datapoint_path = output_dir / run_id
    torch.save(inputs_normed, datapoint_path)
    log.debug(f"Saved processed datapoint: {datapoint_path}")


def run_visualization(
    datasetType: DatasetType, streamlines: dict[str, torch.Tensor], results_path: Path, run_name: str
) -> None:
    """Generates plots for all streamlines."""
    results_path.mkdir(exist_ok=True, parents=True)

    for key, tensor_data in streamlines.items():
        prop_name = expand_property_names(key)[0]
        output_name = results_path / f"{run_name}-{prop_name}"

        visualize_streamlines(datasetType, output_name, prop_name, tensor_data)
