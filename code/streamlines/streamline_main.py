import time
from code.preprocessing.datasets.dataset import DatasetType
from code.preprocessing.preprocessing import preprocessing
from code.preprocessing.transforms import NormalizeTransform
from code.processing.networks.unetVariants import UNet
from code.streamlines.calculation.streamline_calculation import compute_physics_streamlines
from code.streamlines.environment_pre import (
    extract_heat_pump_positions,
    load_velocity_field,
    run_visualization,
    save_result,
)
from code.streamlines.tensor_manipulation import (
    create_property_index_map,
    crop_and_merge_tensors,
    expand_input_dimensions,
    prepare_dataset_step3,
)
from code.utils import logging as log  # noqa: F401
from code.utils.utils_args import get_data_prep_path, load_yaml, make_data_prep_dir
from code.utils.yaml_parser import AppConfig, SimulationStepConfig, convert_injection_config
from pathlib import Path

import matplotlib
import torch
from tqdm import tqdm

# Set backend to Agg for headless production environments
matplotlib.use("Agg")


def process_single_datapoint(
    run_path: Path,
    destination_path: Path,
    origin_data_path: Path,
    results_path: Path,
    step2_config: SimulationStepConfig,
    use_velocity_model: bool,
    model: UNet | None,
    norm_v: NormalizeTransform,
    norm_before: NormalizeTransform,
    norm_after: NormalizeTransform,
    step3_map: dict[str, int],
    step2_map: dict[str, int],
    step1_map: dict[str, int],
    device: torch.device,
) -> None:
    """Orchestrates physics calculation, visualization, and saving for a single simulation run."""
    run_id = run_path.name
    start_time = time.time()

    # 1. Load Data
    velocity_field = load_velocity_field(run_id, origin_data_path, use_velocity_model, model, norm_v, device)
    inputs: torch.Tensor = torch.load(destination_path / "Inputs" / run_id)
    norm_before.reverse(inputs, "Inputs")

    # 2. Geometry Prep
    inputs_merged = crop_and_merge_tensors(
        expand_input_dimensions(inputs, step2_map, step3_map),
        velocity_field,
        step1_map,
        step3_map,
    )

    # 3. Physics Calculation
    pos_hps_np = extract_heat_pump_positions(inputs_merged, step3_map["i"])
    log.info(f"Processing {run_id}: {pos_hps_np.shape[0]} heat pumps detected.")

    streamlines = compute_physics_streamlines(
        step2_config,
        torch.from_numpy(pos_hps_np).float().to(device),
        velocity_field[step1_map["x"]].float().t().to(device),
        velocity_field[step1_map["y"]].float().t().to(device),
        inputs_merged[step3_map["i"]].shape,
    )

    (temp, seasonalCycleSteps) = convert_injection_config(
        step2_config.physical_parameters.injection_temperature_C,
        step2_config.directsolver.steps,
        step2_config.physical_parameters.duration_years,
        device,
    )
    min_temp = temp.min()
    max_temp = temp.max()

    datasetType = None
    if min_temp > 10.6:
        datasetType = DatasetType.steady_state_heating
        log.info(f"Dataset type: Steady State Heating (min temp {min_temp}°C > 10.6°C)")
    elif max_temp <= 10.6:
        datasetType = DatasetType.steady_state_cooling
        log.info(f"Dataset type: Steady State Cooling (max temp {max_temp}°C <= 10.6°C)")
    else:
        datasetType = DatasetType.seasonal
        log.info(f"Dataset type: Seasonal (min temp {min_temp}°C <= 10.6°C < max temp {max_temp}°C)")

    # 4. Visualize & Save
    run_visualization(datasetType, streamlines, results_path, run_path.stem)
    save_result(destination_path, run_id, inputs_merged, streamlines, norm_after, step3_map, step2_map)

    log.info(f"Finished {run_id} in {time.time() - start_time:.2f}s")


def initialize_velocity_model(
    config: AppConfig,
    device: torch.device,
    use_velocity_model: bool,
    origin_data_path: Path,
) -> tuple[UNet | None, Path]:
    """Initializes the UNet model if velocity prediction is required."""
    if not use_velocity_model:
        return None, origin_data_path

    model_dir = config.paths.results / config.run_configuration.run_name / "step1"
    model_config = config.general_configuration.step1.model_parameters

    model = UNet(
        in_channels=len(model_config.inputs),
        out_channels=len(model_config.outputs),
        kernel_size=model_config.kernel_size,
        depth=model_config.depth,
        init_features=model_config.init_features,
        stride=model_config.stride,
        dilation=model_config.dilation,
        activation=model_config.activation,
        norm=model_config.norm,
        repeat_inner=model_config.repeat_inner,
    )
    model.load(model_dir, device, "model.pt")
    model.eval()
    return model, model_dir


def _run_preprocessing(inputs: list[str], outputs: list[str], base: Path, raw: Path, destination_path: Path) -> Path:
    """Helper for repeated preprocessing directory setup and execution."""
    i_str, o_str = "".join(inputs), "".join(outputs)
    target_dir = get_data_prep_path(base, i_str, o_str, raw)
    make_data_prep_dir(target_dir)
    preprocessing(
        {
            "inputs": i_str,
            "outputs": o_str,
            "data_raw": raw,
            "data_prep": target_dir,
            "case": "train",
            "model": None,
            "destination": destination_path,
        }
    )
    return target_dir


def execute_streamline_pipeline(config: AppConfig, mode: str) -> None:
    """Main pipeline entry point."""
    if mode != "run":
        raise ValueError(f"Mode '{mode}' is not implemented. Only 'run' is supported.")

    # Config Setup
    run_conf = config.run_configuration
    gen_conf = config.general_configuration
    dataset_path = config.paths.datasets_raw / run_conf.dataset
    prep_path = config.paths.datasets_prep
    results_path = config.paths.results / run_conf.run_name / "step2"
    results_path.mkdir(parents=True, exist_ok=True)

    # I/O Definitions
    step1_in, step1_out = gen_conf.step1.model_parameters.inputs, gen_conf.step1.model_parameters.outputs
    step2_in, step2_out = step1_in + step1_out, ["t"]
    step3_in, step3_out = gen_conf.step3.model_parameters.inputs, gen_conf.step3.model_parameters.outputs

    # Preprocessing
    step1_dir = _run_preprocessing(step1_in, step1_out, prep_path, dataset_path, results_path)
    step2_dir = _run_preprocessing(step2_in, step2_out, prep_path, dataset_path, results_path)

    # Initialization
    device = torch.device(run_conf.device)
    model, velocity_dir = initialize_velocity_model(config, device, run_conf.use_velocity_model, step1_dir)

    # Dataset Preparation Step 3
    step3_dir = get_data_prep_path(prep_path, "".join(step3_in), "".join(step3_out), dataset_path)
    prepare_dataset_step3(
        step2_dir,
        step3_dir,
        velocity_dir,
        run_conf.use_velocity_model,
        run_conf.dataset,
        step1_in,
        step1_out,
        step2_in,
        step3_in,
        step3_out,
    )

    # Normalization & Mappings
    norm_v = NormalizeTransform(load_yaml(velocity_dir / "info.yaml"))
    norm_before = NormalizeTransform(load_yaml(step2_dir / "info.yaml"))
    norm_after = NormalizeTransform(load_yaml(step3_dir / "info.yaml"))

    step1_outputs_map = create_property_index_map(step1_out)
    step2_inputs_map = create_property_index_map(step2_in)
    step3_inputs_map = create_property_index_map(step3_in)

    # Processing Loop
    inputs_dir = step3_dir / "Inputs"
    if not inputs_dir.exists():
        raise FileNotFoundError(f"Inputs directory missing: {inputs_dir}")

    files = list(inputs_dir.iterdir())
    log.info(f"Starting streamline calculation for {len(files)} items.")
    files.sort(key=lambda x: x.name)

    for run_path in tqdm(files, desc="Calculating Streamlines"):
        process_single_datapoint(
            run_path=run_path,
            destination_path=step3_dir,
            origin_data_path=step1_dir,
            results_path=results_path,
            step2_config=gen_conf.step2,
            use_velocity_model=run_conf.use_velocity_model,
            model=model,
            norm_v=norm_v,
            norm_before=norm_before,
            norm_after=norm_after,
            step3_map=step3_inputs_map,
            step2_map=step2_inputs_map,
            step1_map=step1_outputs_map,
            device=device,
        )
