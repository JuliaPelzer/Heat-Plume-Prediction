from code.preprocessing.preprocessing import expand_property_names
from code.utils import logging as log  # noqa: F401
from code.utils.utils_args import load_yaml, save_yaml
from pathlib import Path
from shutil import copytree
from typing import Any

import torch


def create_property_index_map(property_names: list[str]) -> dict[str, int]:
    """Create a mapping from property name to their tensor indices."""
    return {name: i for i, name in enumerate(property_names)}


def expand_input_dimensions(
    inputs_normed: torch.Tensor, src_map: dict[str, int], tgt_map: dict[str, int]
) -> torch.Tensor:
    """Reshape input tensor to target config; maps existing channels and zeros new ones."""
    out = torch.zeros((len(tgt_map), *inputs_normed.shape[1:]), dtype=inputs_normed.dtype, device=inputs_normed.device)
    for name, src_idx in src_map.items():
        if name in tgt_map:
            out[tgt_map[name]] = inputs_normed[src_idx]
    return out.float()


def prepare_dataset_step3(
    step2_dir: Path,
    dest_dir: Path,
    vel_model_dir: Path,
    use_vel_model: bool,
    dataset_name: str,
    step1_inputs: list[str],
    step1_outputs: list[str],
    step2_inputs: list[str],
    step3_inputs: list[str],
    step3_outputs: list[str],
) -> None:
    """Consolidate metadata, propagate normalization stats, and generate args.yaml."""
    copytree(step2_dir, dest_dir, dirs_exist_ok=True)

    # Load configuration files
    info = load_yaml(dest_dir / "info.yaml")
    vel_info = load_yaml(vel_model_dir / "info.yaml")

    # Pre-load available statistics from Step 2 Inputs and Velocity Outputs
    stats_pool: dict[str, Any] = {}

    # 1. Preserve stats from Step 2 (using Step 1 names for lookup compatibility)
    for key in step1_inputs:
        expanded = expand_property_names(key)[0]
        if expanded in info.get("Inputs", {}):
            stats_pool[key] = info["Inputs"][expanded]

    # 2. Capture stats from Velocity Model Outputs
    for key in step1_outputs:
        expanded = expand_property_names(key)[0]
        if expanded in vel_info.get("Labels", {}):
            stats_pool[key] = vel_info["Labels"][expanded]

    # Rebuild Inputs section in destination info.yaml
    info["Inputs"] = {}
    step3_map = create_property_index_map(step3_inputs)

    # Collect expanded names for args.yaml
    expanded_step3_inputs = []

    for idx, key in enumerate(step3_inputs):
        props = expand_property_names(key)
        expanded_step3_inputs.extend(props)
        prop_name = props[0]  # Use primary name for stats

        # Assign stats: Use pool if available, else default synthetic values
        if key in stats_pool:
            info["Inputs"][prop_name] = stats_pool[key]
        else:
            info["Inputs"][prop_name] = {"max": 1.0, "mean": None, "min": 0.0, "norm": None, "std": None}
        info["Inputs"][prop_name]["index"] = idx

    save_yaml(info, dest_dir / "info.yaml")

    # Generate args.yaml
    args = {
        "dataset": dataset_name,
        "inputs": expanded_step3_inputs,
        "outputs": [n for k in step3_outputs for n in expand_property_names(k)],
    }

    if use_vel_model:
        # Tag inputs predicted by velocity model
        for key in step1_outputs:
            if key in step3_map:
                args["inputs"][step3_map[key]] += f" - predicted by '{vel_model_dir.name}'"

    save_yaml(args, dest_dir / "args.yaml")


def crop_and_merge_tensors(
    inputs: torch.Tensor, velocity: torch.Tensor, vel_out_map: dict[str, int], step3_in_map: dict[str, int]
) -> torch.Tensor:
    """Center-crop inputs to match velocity dimensions and inject velocity channels."""
    _, h_req, w_req = velocity.shape
    _, h_curr, w_curr = inputs.shape

    # Calculate center crop offsets
    dy, dx = (h_curr - h_req) // 2, (w_curr - w_req) // 2

    # Clone to decouple memory and perform crop
    merged = inputs[:, dy : dy + h_req, dx : dx + w_req].clone()

    for name, src_idx in vel_out_map.items():
        if name in step3_in_map:
            merged[step3_in_map[name]] = velocity[src_idx]

    return merged
