from code.streamlines.calculation.streamline_rwpt import run_rwpt_thermal_prior
from code.streamlines.calculation.streamline_tensors import make_streamlines_gpu
from code.utils import logging as log  # noqa: F401
from code.utils.yaml_parser import SimulationStepConfig
from typing import Any

import torch


def compute_physics_streamlines(
    step2_config: SimulationStepConfig,
    heat_pump_pos: torch.Tensor,
    vx_m_per_year: torch.Tensor,
    vy_m_per_year: torch.Tensor,
    dims: Any,
) -> dict[str, torch.Tensor]:
    """Compute stochastic streamline heatmaps (1–6) and RWPT thermal prior (7)."""

    (
        stream_sum_pos,
        stream_uncertainty,
        stream_faded_pos,
        stream_max_faded,
        stream_max_seasons,
        stream_sum_seasons_pos,
    ) = make_streamlines_gpu(
        step2_config.streamlines,
        step2_config.physical_parameters,
        heat_pump_pos.clone(),
        vx_m_per_year,
        vy_m_per_year,
        dims,
    )

    thermal_prior: torch.Tensor = run_rwpt_thermal_prior(
        step2_config.rwpt,
        step2_config.physical_parameters,
        heat_pump_pos.clone(),
        vx_m_per_year,
        vy_m_per_year,
        dims,
    )

    # Dictionary keys map to config property names
    return {
        "1": stream_sum_pos,
        "2": stream_uncertainty,
        "3": stream_faded_pos,
        "4": stream_max_faded,
        "5": stream_sum_seasons_pos,
        "6": stream_max_seasons,
        "7": thermal_prior,
    }
