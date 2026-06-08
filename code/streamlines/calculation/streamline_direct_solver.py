from code.utils import logging as log  # noqa: F401
from code.utils.yaml_parser import convert_injection_config
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# --- PHYSICAL CONSTANTS (SI UNITS) ---
secondsPerYear_s = 365.25 * 24 * 3600


@dataclass
class SimulationConfig:
    """
    Configuration for the Seasonal Steady-State Heat Plume Simulation.
    """

    device: str
    resolution_m_per_px: float
    ambientTemp_C: float

    # Injection Data, injectionRates: [Sources, TimeSteps]
    injectionRate_m3_per_s: torch.Tensor
    injectionTemp_C: torch.Tensor

    # Simulation Timeline
    timeSteps_count: int
    timeEnd_years: float
    seasonalCycleSteps: int

    # Sampling: Number of particles released per source over one seasonal cycle
    samplesPerSource_count: int

    # Aquifer Properties
    porosity_frac: float
    rockDensity_kg_per_m3: float
    rockSpecificHeat_J_per_kgK: float
    waterDensity_kg_per_m3: float
    waterSpecificHeat_J_per_kgK: float
    thermalConductivityDry_W_per_mK: float
    thermalConductivityWet_W_per_mK: float
    thicknessAquifer_m: float

    # Dispersion
    longitudinalDispersivity_m: float
    transverseDispersivityH_m: float

    @property
    def timeStep_s(self) -> float:
        """Returns the time step in Seconds."""
        return (self.timeEnd_years * secondsPerYear_s) / self.timeSteps_count

    @property
    def molecularDiffusion_m2_per_s(self) -> float:
        """Calculates effective molecular diffusion D_m [m^2/s]."""
        return self.thermalConductivityWet_W_per_mK / (
            self.porosity_frac * self.waterDensity_kg_per_m3 * self.waterSpecificHeat_J_per_kgK
        )

    @property
    def retardationFactor_dimless(self) -> float:
        """Calculates Thermal Retardation Factor (R)."""
        volumetricHeatRatio_dimless = (self.rockDensity_kg_per_m3 * self.rockSpecificHeat_J_per_kgK) / (
            self.waterDensity_kg_per_m3 * self.waterSpecificHeat_J_per_kgK
        )
        return 1.0 + ((1.0 - self.porosity_frac) / self.porosity_frac) * volumetricHeatRatio_dimless


# --- JIT COMPILED PHYSICS KERNELS ---
@torch.jit.script
def sample_velocity_bilinear(
    positions_px: torch.Tensor,
    fieldTensor_m_per_year: torch.Tensor,
    invWidth_per_px: float,
    invHeight_per_px: float,
    resolution_m_per_px: float,
    secondsPerYear_s: float,
) -> torch.Tensor:
    """
    Samples the background velocity field (m/year) at particle positions.
    Returns velocity in px/s.
    """
    N = positions_px.size(0)
    # Map px coordinates to normalized grid [-1, 1]
    nx = (positions_px[:, 0] * invWidth_per_px - 1.0).clamp(-1.0, 1.0)
    ny = (positions_px[:, 1] * invHeight_per_px - 1.0).clamp(-1.0, 1.0)
    grid = torch.stack((nx, ny), dim=-1).view(1, N, 1, 2)

    # Bilinear Interpolation
    vSample_m_per_year = F.grid_sample(
        fieldTensor_m_per_year, grid, mode="bilinear", padding_mode="border", align_corners=True
    )

    # Convert (m/year) -> (m/s) -> (px/s)
    return (vSample_m_per_year.view(2, N).t() / secondsPerYear_s) / resolution_m_per_px


@torch.jit.script
def rwpt_seasonal_stream_kernel(
    accumEnergyGrid_flat: torch.Tensor,
    positions_px: torch.Tensor,
    origins_px: torch.Tensor,
    fieldTensor_m_per_year: torch.Tensor,
    injectionRates_m3_per_s: torch.Tensor,
    sourceIndices_idx: torch.Tensor,
    injectionWeights_full: torch.Tensor,
    birthIndices: torch.Tensor,
    timeStep_s: float,
    steps_count: int,
    cycle_steps: int,
    width_px: int,
    height_px: int,
    resolution_m_per_px: float,
    alphaL_m: float,
    alphaT_m: float,
    molecularDiffusion_m2_per_s: float,
    porosity_frac: float,
    retardationFactor_dimless: float,
    secondsPerYear_s: float,
    thicknessAquifer_m: float,
):
    """
    Simulates particle transport and rasterizes their pathlines into the grid
    in a single pass (Stream Rasterization). Handles seasonal cyclic injection.
    """
    N = positions_px.size(0)
    device = positions_px.device

    # Precompute constants
    invWidth_per_px = 2.0 / (width_px - 1.0)
    invHeight_per_px = 2.0 / (height_px - 1.0)
    sqrt2Dt_sqrt_s = torch.sqrt(torch.tensor(2.0 * timeStep_s, device=device))

    geometryConst_per_m2 = 1.0 / (2.0 * torch.pi * thicknessAquifer_m * porosity_frac * resolution_m_per_px**2)
    thermalRetardationScale_dimless = 1.0 / retardationFactor_dimless
    grid_h = accumEnergyGrid_flat.size(0) // width_px

    # Initialize previous position (start at origin)
    # We copy origins to positions to ensure clean start state
    positions_px.copy_(origins_px)
    prev_pos = origins_px.clone()

    for i in range(steps_count):
        # Background Velocity (px/s)
        qBackground_px_per_s = sample_velocity_bilinear(
            positions_px,
            fieldTensor_m_per_year,
            invWidth_per_px,
            invHeight_per_px,
            resolution_m_per_px,
            secondsPerYear_s,
        )

        # --- 1. MASKING ---
        # Particles are active if current simulation step >= birth step
        active_sim_mask = i >= birthIndices

        # --- 2. PHYSICS (SEASONAL) ---
        # Cyclic time lookup: t = i % steps_per_year
        seasonal_t = i % cycle_steps

        # Get Rate for current seasonal phase
        currentQ_m3_per_s = injectionRates_m3_per_s[sourceIndices_idx, seasonal_t].unsqueeze(1)

        # Radial Velocity (px/s)
        deltaPos_px = positions_px - origins_px
        distSq_px2 = deltaPos_px.pow(2).sum(dim=1, keepdim=True)
        dist_px = distSq_px2.sqrt() + 1e-4

        vRadMag_px_per_s = (currentQ_m3_per_s * geometryConst_per_m2) / dist_px
        dirRad_dimless = deltaPos_px / dist_px
        qTotal_px_per_s = qBackground_px_per_s + dirRad_dimless * (vRadMag_px_per_s / secondsPerYear_s)

        # Seepage Velocity & Dispersion Magnitude
        vSeepage_px_per_s = qTotal_px_per_s / porosity_frac
        vSeepageMag_px_per_s = vSeepage_px_per_s.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-12
        vThermal_px_per_s = vSeepage_px_per_s * thermalRetardationScale_dimless

        # Dispersion Tensors
        uL_dimless = vSeepage_px_per_s / vSeepageMag_px_per_s
        uT_dimless = torch.stack([-uL_dimless[:, 1], uL_dimless[:, 0]], dim=1)

        diffusionL_px2_per_s = (
            (molecularDiffusion_m2_per_s + alphaL_m * vSeepageMag_px_per_s) / retardationFactor_dimless
        ) / resolution_m_per_px
        diffusionT_px2_per_s = (
            (molecularDiffusion_m2_per_s + alphaT_m * vSeepageMag_px_per_s) / retardationFactor_dimless
        ) / resolution_m_per_px

        stepL_px = torch.randn((N, 1), device=device) * torch.sqrt(diffusionL_px2_per_s) * sqrt2Dt_sqrt_s
        stepT_px = torch.randn((N, 1), device=device) * torch.sqrt(diffusionT_px2_per_s) * sqrt2Dt_sqrt_s

        # Random Walk Step
        displacement_px = (vThermal_px_per_s * timeStep_s) + (uL_dimless * stepL_px) + (uT_dimless * stepT_px)

        # Update Position (Masked)
        mask_float = active_sim_mask.view(N, 1).float()
        positions_px += displacement_px * mask_float

        # --- KINEMATIC SNAPSHOT MAPPING ---
        # Determine the phase of the water currently residing at this pathline segment
        age_steps = i - birthIndices
        T_final = steps_count - 1
        phase_steps = (T_final - age_steps) % cycle_steps

        # Lookup the exact physical weight for this spatial location
        w_snap = injectionWeights_full[sourceIndices_idx, phase_steps]

        # Filter inactive particles or near-zero injection anomalies
        draw_mask = active_sim_mask & (w_snap.abs() > 1e-12)

        if draw_mask.any():
            p1 = prev_pos[draw_mask]
            p2 = positions_px[draw_mask]
            # Scale the weight by the current seasonal intensity
            w_draw = w_snap[draw_mask]

            diff = p2 - p1
            dist_seg = (diff.pow(2).sum(dim=1)).sqrt()
            n_steps = torch.ceil(dist_seg * 1.5).long().clamp(min=1)

            # Vectorized Expansion
            p1_exp = torch.repeat_interleave(p1, n_steps, dim=0)
            diff_exp = torch.repeat_interleave(diff, n_steps, dim=0)
            w_exp = torch.repeat_interleave(w_draw, n_steps, dim=0)
            n_steps_exp = torch.repeat_interleave(n_steps, n_steps, dim=0)

            # Generate Alpha [0, 1/N, 2/N...]
            seg_ends = torch.cumsum(n_steps, dim=0)
            seg_starts = torch.cat((torch.zeros(1, device=device, dtype=torch.long), seg_ends[:-1]))
            seg_starts_exp = torch.repeat_interleave(seg_starts, n_steps, dim=0)

            global_idx = torch.arange(p1_exp.size(0), device=device)
            local_idx = global_idx - seg_starts_exp

            alpha = local_idx.float() / n_steps_exp.float()

            # Interpolate
            pos_x = p1_exp[:, 0] + diff_exp[:, 0] * alpha
            pos_y = p1_exp[:, 1] + diff_exp[:, 1] * alpha

            # Distribute Weight (Energy / Steps)
            w_dot = w_exp / n_steps_exp.float()

            # Boundary Check & Accumulate
            valid_x = pos_x.long()
            valid_y = pos_y.long()

            valid_map = (valid_x >= 0) & (valid_x < width_px) & (valid_y >= 0) & (valid_y < grid_h)

            if valid_map.any():
                flat_idx = valid_y[valid_map] * width_px + valid_x[valid_map]
                accumEnergyGrid_flat.scatter_add_(0, flat_idx, w_dot[valid_map])

        # Update previous position
        prev_pos.copy_(positions_px)


# --- MAIN DRIVER ---


def generate_physical_plumes(
    config: SimulationConfig,
    heatPumpPositions_px: torch.Tensor,
    vx_m_per_year: torch.Tensor,
    vy_m_per_year: torch.Tensor,
    dims_px: tuple[int, int],
) -> torch.Tensor:
    """
    Main entry point for generating heat plumes using RWPT with seasonal logic.
    """
    device = torch.device(config.device)
    gridWidth_px, gridHeight_px = dims_px
    numHps_count = len(heatPumpPositions_px)

    with torch.inference_mode():
        # 1. Prepare Velocity Field
        vField_m_per_year = torch.stack([vy_m_per_year, vx_m_per_year]).unsqueeze(0).to(device)

        # 2. Prepare Globals
        globalEnergy_J = torch.zeros((gridHeight_px * gridWidth_px), device=device)

        # Extract just one cycle of rates for the kernel
        cycle_len = config.seasonalCycleSteps

        # 3. Simulation Loop (Batched by Source)
        hpBatchSize_count = 10

        for bIdx in tqdm(range(0, numHps_count, hpBatchSize_count), "Direct Solver"):
            bEnd = min(bIdx + hpBatchSize_count, numHps_count)
            batchPos = heatPumpPositions_px[bIdx:bEnd]
            currBatchSize = bEnd - bIdx

            # --- Stratified Sampling (Year 0) ---
            # We release particles evenly distributed over the first year (cycle)
            cycle_times = torch.linspace(0, cycle_len - 1, config.samplesPerSource_count, device=device).long()

            birthIndices = cycle_times.unsqueeze(0).expand(currBatchSize, -1)
            srcIndices = (
                torch.arange(currBatchSize, device=device).unsqueeze(1).expand(-1, config.samplesPerSource_count)
            )

            # Precompute full weight matrix [Sources, Cycle_Steps]
            pRates = config.injectionRate_m3_per_s[bIdx:bEnd]
            pTemps = config.injectionTemp_C[bIdx:bEnd]
            injectionWeights_full = pRates * (pTemps - config.ambientTemp_C)

            # Flatten for Kernel
            flat_birth = birthIndices.flatten()
            flat_src = srcIndices.flatten()

            flat_origins = batchPos.repeat_interleave(config.samplesPerSource_count, dim=0)

            # Initialize Particles with slight spread
            radius = 0.15 / config.resolution_m_per_px
            flat_particles = flat_origins.clone() + torch.randn_like(flat_origins) * radius

            # Run Kernel
            rwpt_seasonal_stream_kernel(
                globalEnergy_J,
                flat_particles,
                flat_origins,
                vField_m_per_year,
                config.injectionRate_m3_per_s[bIdx:bEnd],
                flat_src,
                injectionWeights_full,
                flat_birth,
                float(config.timeStep_s),
                config.timeSteps_count,
                cycle_len,
                gridWidth_px,
                gridHeight_px,
                float(config.resolution_m_per_px),
                config.longitudinalDispersivity_m,
                config.transverseDispersivityH_m,
                config.molecularDiffusion_m2_per_s,
                config.porosity_frac,
                config.retardationFactor_dimless,
                secondsPerYear_s,
                config.thicknessAquifer_m,
            )

    # --- POST-PROCESS: BLUR & SCALE ---

    # 1. Gaussian Blur (Smoothing)
    # Ensures the filament lines blend into a smooth gradient
    globalEnergy2d_J = globalEnergy_J.view(1, 1, gridHeight_px, gridWidth_px)
    kernel_size = 5
    sigma = 3.0
    k = torch.tensor(
        [np.exp(-0.5 * (x - kernel_size // 2) ** 2 / sigma**2) for x in range(kernel_size)],
        device=device,
        dtype=torch.float32,
    )
    k = k / k.sum()
    kernel = k[:, None] * k[None, :]
    kernel = kernel.expand(1, 1, kernel_size, kernel_size)

    pad_size = kernel_size // 2
    globalEnergyPadded = F.pad(globalEnergy2d_J, (pad_size, pad_size, pad_size, pad_size), mode="replicate")
    globalEnergyBlurred_J = F.conv2d(globalEnergyPadded, kernel, padding=0).squeeze()

    # 2. Scaling (Steady State)
    # Temp = Integral(Flux * dt) / (Vol * RhoC * Samples)
    volCell_m3 = (config.resolution_m_per_px**2) * config.thicknessAquifer_m
    rhoCw = config.waterDensity_kg_per_m3 * config.waterSpecificHeat_J_per_kgK
    rhoCaq = (config.porosity_frac * rhoCw) + (
        (1 - config.porosity_frac) * config.rockDensity_kg_per_m3 * config.rockSpecificHeat_J_per_kgK
    )

    scalingFactor_C_per_count = (config.timeStep_s * rhoCw) / (volCell_m3 * rhoCaq * config.samplesPerSource_count)

    finalTempMap_C = config.ambientTemp_C + (globalEnergyBlurred_J * scalingFactor_C_per_count)

    # 3. Smart Clamp
    global_min = min(config.injectionTemp_C.min().item(), config.ambientTemp_C)
    global_max = max(config.injectionTemp_C.max().item(), config.ambientTemp_C)

    finalTempMap_C = torch.clamp(finalTempMap_C, min=global_min, max=global_max)

    return finalTempMap_C.t().cpu()


def direct_solve(
    modeDirectSolver: Any,
    modeConstants: Any,
    heatPumpPositions_px: torch.Tensor,
    vx_m_per_year: torch.Tensor,
    vy_m_per_year: torch.Tensor,
    dims_px: tuple[int, int],
) -> torch.Tensor:
    """
    Setup and run the simulation.
    """
    numHps_count = len(heatPumpPositions_px)

    # --- CONFIGURATION ---
    PFLOTRAN_SCALING_FACTOR = 0.25
    (rate, seasonalCycleSteps1) = convert_injection_config(
        modeConstants.injection_rate_m3_per_s,
        modeDirectSolver.steps,
        modeConstants.duration_years,
        heatPumpPositions_px.device,
    )
    (temp, seasonalCycleSteps2) = convert_injection_config(
        modeConstants.injection_temperature_C,
        modeDirectSolver.steps,
        modeConstants.duration_years,
        heatPumpPositions_px.device,
    )
    assert seasonalCycleSteps1 == seasonalCycleSteps2, "Mismatch in seasonal cycle steps between rate and temperature"
    seasonalCycleSteps = seasonalCycleSteps1

    simConfig = SimulationConfig(
        device=heatPumpPositions_px.device,
        # props
        samplesPerSource_count=modeDirectSolver.samples,
        timeSteps_count=modeDirectSolver.steps,
        # consts
        resolution_m_per_px=modeConstants.resolution_m,
        ambientTemp_C=modeConstants.ambient_temperature_C,
        injectionRate_m3_per_s=rate.unsqueeze(0).repeat(numHps_count, 1) * PFLOTRAN_SCALING_FACTOR,
        injectionTemp_C=temp.unsqueeze(0).repeat(numHps_count, 1),
        timeEnd_years=modeConstants.duration_years,
        seasonalCycleSteps=seasonalCycleSteps,
        porosity_frac=modeConstants.porosity_frac,
        rockDensity_kg_per_m3=modeConstants.rock_density_kg_per_m3,
        rockSpecificHeat_J_per_kgK=modeConstants.rock_specific_heat_J_per_kgK,
        waterDensity_kg_per_m3=modeConstants.water_density_kg_per_m3,
        waterSpecificHeat_J_per_kgK=modeConstants.water_specific_heat_J_per_kgK,
        thermalConductivityDry_W_per_mK=modeConstants.thermal_conductivity_dry_W_per_mK,
        thermalConductivityWet_W_per_mK=modeConstants.thermal_conductivity_wet_W_per_mK,
        thicknessAquifer_m=modeConstants.thickness_aquifer_m,
        longitudinalDispersivity_m=modeConstants.longitudinal_dispersivity_m,
        transverseDispersivityH_m=modeConstants.transverse_dispersivity_h_m,
    )

    # --- EXECUTE ---
    tMap_normalized = generate_physical_plumes(simConfig, heatPumpPositions_px, vx_m_per_year, vy_m_per_year, dims_px)

    debug = False
    if debug:
        log.info("DEBUG MODE: Cropping and Rescaling Output")
        x = 0
        tMap_normalized = tMap_normalized[x : 1000 - x, x : 1000 - x]
        tMap_normalized = F.interpolate(
            tMap_normalized.unsqueeze(0).unsqueeze(0), size=(1000, 1000), mode="bilinear", align_corners=False
        ).squeeze()
    else:
        log.info("Release Mode: Rescaling Output to [0, 1]")

        temp_spread = modeConstants.temperature_spread_C
        if 10.6 <= tMap_normalized.min().item():
            tMap_normalized = (tMap_normalized - modeConstants.ambient_temperature_C) / temp_spread
        elif tMap_normalized.max().item() <= 10.6:
            tMap_normalized = (tMap_normalized - (modeConstants.ambient_temperature_C - temp_spread)) / temp_spread
        else:
            tMap_normalized = (tMap_normalized - (modeConstants.ambient_temperature_C - temp_spread)) / (
                temp_spread * 2
            )

    log.info(f"Min: {tMap_normalized.min():.2f}, Max: {tMap_normalized.max():.2f}, Mean: {tMap_normalized.mean():.4f}")
    return tMap_normalized
