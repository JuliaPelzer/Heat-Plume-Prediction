from code.utils import logging as log  # noqa: F401
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar

import numpy as np
import torch
import yaml
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError

# --- Helper Types & Validators ---


def force_string(v: Any) -> str:
    """Coerces any input to string."""
    return str(v)


CoercedString = Annotated[str, BeforeValidator(force_string)]
T = TypeVar("T", bound=BaseModel)

# --- Configuration Models ---


class Datapoints(BaseModel):
    """Defines split sizes for datasets."""

    validation: list[int]
    test: list[int]
    train: list[int]


class GeneralSettings(BaseModel):
    """General training settings."""

    epochs: int
    visualize: bool


class ReduceLROnPlateauConfig(BaseModel):
    """Configuration for ReduceLROnPlateau scheduler."""

    type: Literal["ReduceLROnPlateau"]
    init_lr: float
    mode: str = "min"
    factor: float
    patience: int
    threshold: float
    min_lr: float


class StepLRConfig(BaseModel):
    """Configuration for StepLR scheduler."""

    type: Literal["StepLR"]
    init_lr: float
    step_size: int
    gamma: float


SchedulerConfig = Annotated[ReduceLROnPlateauConfig | StepLRConfig, Field(discriminator="type")]


class UNetParameters(BaseModel):
    """Hyperparameters specific to UNet architecture."""

    network: Literal["unet"]
    inputs: list[CoercedString]
    outputs: list[CoercedString]
    batchsize: int
    kernel_size: int
    depth: int
    init_features: int
    stride: int
    dilation: int
    norm: str | None
    repeat_inner: bool
    skip_per_dir: int
    len_box: int
    train_loss: str
    bool_cutouts: bool
    optimizer_switch: bool
    optimizer: str
    activation: str


class RNNParameters(BaseModel):
    """Hyperparameters specific to RNN architecture."""

    model_config = ConfigDict(extra="allow")  # TODO @Johanna: Allow dynamic fields for incomplete implementation
    network: Literal["rnn"]
    inputs: list[CoercedString]
    outputs: list[CoercedString]


ModelConfig = Annotated[UNetParameters | RNNParameters, Field(discriminator="network")]


class HoptParameters(BaseModel):
    """Search space for hyperparameter optimization."""

    network: list[str]
    inputs: list[list[CoercedString]]
    outputs: list[list[CoercedString]]
    batchsize: list[int]
    kernel_size: list[int]
    depth: list[int]
    init_features: list[int]
    stride: list[int]
    dilation: list[int]
    norm: list[str | None]
    repeat_inner: list[bool]
    skip_per_dir: list[int]
    len_box: list[int]
    train_loss: list[str]
    bool_cutouts: list[bool]
    optimizer_switch: list[bool]
    optimizer: list[str]
    activation: list[str]


class MLStepConfig(BaseModel):
    """Configuration for ML pipeline steps."""

    datapoints: Datapoints
    general: GeneralSettings
    scheduler: SchedulerConfig
    model_parameters: ModelConfig
    hopt_parameters: HoptParameters | None = None


class Streamlines(BaseModel):
    """Streamline simulation parameters."""

    samples: int
    steps: int
    diffusion_scale: float
    diffusion_base: float


class DirectSolver(BaseModel):
    """Direct solver parameters."""

    samples: int
    steps: int


class TimeSeriesParam(BaseModel):
    """Time-dependent physical parameters."""

    time_unit: str
    values: dict[float, float]


class PhysicalParameters(BaseModel):
    """Physical constants and field parameters."""

    resolution_m: float
    duration_years: float
    porosity_frac: float
    rock_density_kg_per_m3: float
    rock_specific_heat_J_per_kgK: float
    water_density_kg_per_m3: float
    water_specific_heat_J_per_kgK: float
    thermal_conductivity_dry_W_per_mK: float
    thermal_conductivity_wet_W_per_mK: float
    thickness_aquifer_m: float
    longitudinal_dispersivity_m: float
    transverse_dispersivity_h_m: float
    ambient_temperature_C: float
    temperature_spread_C: float
    injection_temperature_C: TimeSeriesParam
    injection_rate_m3_per_s: TimeSeriesParam


class SimulationStepConfig(BaseModel):
    """Configuration for physics simulation step."""

    streamlines: Streamlines
    directsolver: DirectSolver
    physical_parameters: PhysicalParameters


class GeneralConfiguration(BaseModel):
    """Aggregated configuration for all pipeline steps."""

    step1: MLStepConfig
    step2: SimulationStepConfig
    step3: MLStepConfig


class RunConfiguration(BaseModel):
    """Meta-configuration for the execution environment."""

    run_name: str
    dataset: str
    seed: int
    device: str
    use_velocity_model: bool
    pipeline: list[dict[str, str]]


class Paths(BaseModel):
    """Directory paths for data and results."""

    datasets_raw: Path
    datasets_prep: Path
    results: Path


class AppConfig(BaseModel):
    """Root configuration object."""

    run_configuration: RunConfiguration
    general_configuration: GeneralConfiguration
    paths: Paths


# --- Logic & Parsers ---
def load_config[T: BaseModel](yaml_path: str | Path, model_cls: type[T]) -> T:
    """Generic loader for YAML to Pydantic models."""
    try:
        with open(yaml_path) as f:
            return model_cls(**yaml.safe_load(f))
    except (ValidationError, FileNotFoundError, yaml.YAMLError) as e:
        raise ValueError(f"Failed to load {model_cls.__name__} from {yaml_path}: {e}") from e


def parse_config(yaml_file_path: str) -> AppConfig:
    """Parses main application configuration."""
    return load_config(yaml_file_path, AppConfig)


def parse_model_config(yaml_file_path: str) -> ModelConfig:
    """Parses specific model configuration."""
    return load_config(yaml_file_path, ModelConfig)


def parse_model_config_unet(yaml_file_path: str) -> UNetParameters:
    """Parses specific model configuration."""
    return load_config(yaml_file_path, UNetParameters)


def convert_injection_config(
    data: TimeSeriesParam, steps_count: int, time_end_years: float, device: str | torch.device
) -> tuple[torch.Tensor, int]:
    """Interpolates time-series injection data into a tensor for simulation."""
    if data.time_unit != "year":
        raise ValueError(f"Unsupported time unit: {data.time_unit}. Only 'year' is supported.")
    if steps_count <= 1:
        raise ValueError("Step count must be greater than 1 to define time intervals.")
    if time_end_years <= 0:
        raise ValueError(f"Time end must be positive. Got {time_end_years}.")

    # Sort control points by time
    sorted_times = np.array(sorted(data.values.keys()))
    sorted_values = np.array([data.values[t] for t in sorted_times])

    if sorted_times[0] != 0.0:
        raise ValueError("Time series must start at time 0.0 years.")
    if sorted_times[-1] != 1.0:
        raise ValueError("Time series must end at time 1.0 years.")

    num_cycles = int(np.ceil(time_end_years))

    base_t = sorted_times[:-1]
    base_v = sorted_values[:-1]

    cycle_offsets = np.arange(num_cycles)

    full_times = (base_t[None, :] + cycle_offsets[:, None]).flatten()
    full_values = np.tile(base_v, num_cycles)

    full_times = np.append(full_times, num_cycles)
    full_values = np.append(full_values, sorted_values[-1])

    # Vectorized interpolation
    t_eval = np.linspace(0, time_end_years, steps_count)
    cycle_values = np.interp(t_eval, full_times, full_values)

    # Calculate seasonal cycles
    steps_per_year = int((steps_count - 1) / time_end_years)

    return torch.from_numpy(cycle_values).float().to(device), steps_per_year
