import time
from code.utils import logging as log  # noqa: F401
from code.utils.logging import info
from code.utils.yaml_parser import convert_injection_config
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm


class TimerGPU:
    """Context manager for high-precision GPU timing."""

    def __init__(self, name: str = "Block"):
        self.name = name
        self.start = 0.0
        self.interval = 0.0

    def __enter__(self) -> "TimerGPU":
        torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        torch.cuda.synchronize()
        self.interval = time.perf_counter() - self.start
        if self.interval < 1e-3:
            t_str = f"{self.interval * 1e6:.2f} µs"
        elif self.interval < 1.0:
            t_str = f"{self.interval * 1e3:.2f} ms"
        else:
            t_str = f"{self.interval:.4f} s"
        info(f"[{self.name}] finished in {t_str}")


def draw_batched_polylines(
    density_accumulator: torch.Tensor,
    faded_density_accumulator: torch.Tensor,
    time_max_accumulator: torch.Tensor,
    seasonal_max_accumulator: torch.Tensor,
    seasonal_density_accumulator_heating: torch.Tensor,
    seasonal_density_accumulator_cooling: torch.Tensor,
    variance_accumulator: torch.Tensor,
    points: torch.Tensor,
    weights_normal: torch.Tensor,
    weights_faded: torch.Tensor,
    weights_seasonal_heating: torch.Tensor,
    weights_seasonal_cooling: torch.Tensor,
    dims: tuple[int, int],
) -> None:
    """
    Optimized GPU rasterizer for batch curve segments using scatter-gather operations.
    Maps individual curve segments to pixels without expanding memory quadratically.

    Args:
      density_accumulator: Global heatmap for standard density (H, W).
      faded_density_accumulator: Global heatmap for time-faded density (H, W).
      time_max_accumulator: Global map tracking max time influence (H, W).
      seasonal_max_accumulator: Global map tracking max seasonal influence (H, W).
      seasonal_density_accumulator_heating: Global heatmap for seasonal heating flow (H, W).
      seasonal_density_accumulator_cooling: Global heatmap for seasonal cooling flow (H, W).
      variance_accumulator: Global heatmap for variance (H, W).
      points: Tensor of point coordinates (Batch, Num_Points, 2).
      weights_normal: 1D Time-dependent weights for standard density.
      weights_faded: 1D Time-dependent weights for faded density.
      weights_seasonal_heating: 1D Time-dependent weights for seasonal heating density.
      weights_seasonal_cooling: 1D Time-dependent weights for seasonal cooling density.
      dims: tuple of (width, height).
    """
    canvas_width, canvas_height = dims
    device = density_accumulator.device

    # Process in chunks to prevent OOM errors on large batches
    BATCH_CHUNK_SIZE = 256

    num_samples, num_points, _ = points.shape
    num_segments = num_points - 1
    spatial_dim = canvas_width * canvas_height

    # Buffers to accumulate the sum of all chunks before averaging
    total_density = torch.zeros_like(density_accumulator)
    total_density_sq = torch.zeros_like(density_accumulator)
    total_faded = torch.zeros_like(faded_density_accumulator)
    total_seasonal_heating = torch.zeros_like(seasonal_density_accumulator_heating)
    total_seasonal_cooling = torch.zeros_like(seasonal_density_accumulator_cooling)

    for b_start in range(0, num_samples, BATCH_CHUNK_SIZE):
        b_end = min(b_start + BATCH_CHUNK_SIZE, num_samples)
        current_batch_size = b_end - b_start

        # --- 1. Geometry Prep ---
        points_chunk = points[b_start:b_end]
        # Add 0.5 for pixel center alignment
        points_centered = points_chunk.view(-1, 2).add(0.5)

        # Flatten to (Batch * Num_Segments, 2)
        flat_points = points_centered.view(current_batch_size, num_points, 2)
        segment_start = flat_points[:, :-1].reshape(-1, 2)
        segment_end = flat_points[:, 1:].reshape(-1, 2)

        # Calculate Chebyshev distance to determine pixels per segment
        delta = segment_end - segment_start
        steps = delta.abs().max(dim=1).values.ceil().long().clamp_min(1)
        inv_steps = 1.0 / steps.float()

        total_pixels = steps.sum().item()
        if total_pixels == 0:
            continue

        # --- 2. Vectorized Index Mapping (Scatter-Gather) ---
        # Determine segment boundaries in the flat pixel array
        segment_boundaries = steps.cumsum(0)

        # Create global indices for every pixel to be drawn
        global_pixel_indices = torch.arange(total_pixels, device=device)

        # Map every pixel back to its segment index
        segment_indices = torch.bucketize(global_pixel_indices, segment_boundaries, right=True)

        # Calculate interpolation factor 't' [0, 1]
        segment_starts_flat = segment_boundaries[segment_indices] - steps[segment_indices]
        t_factor = (global_pixel_indices - segment_starts_flat).float() * inv_steps[segment_indices]

        # --- 3. Geometry Interpolation ---
        # p(t) = p0 + (p1 - p0) * t
        interpolated_coords = segment_start[segment_indices] + delta[segment_indices] * t_factor.unsqueeze(1)

        # --- 4. Weight Interpolation ---
        # Prepare start/end weights for batch
        w_norm_start = weights_normal[:-1].repeat(current_batch_size)
        w_norm_end = weights_normal[1:].repeat(current_batch_size)
        w_faded_start = weights_faded[:-1].repeat(current_batch_size)
        w_faded_end = weights_faded[1:].repeat(current_batch_size)
        w_seas_heating_start = weights_seasonal_heating[:-1].repeat(current_batch_size)
        w_seas_heating_end = weights_seasonal_heating[1:].repeat(current_batch_size)
        w_seas_cooling_start = weights_seasonal_cooling[:-1].repeat(current_batch_size)
        w_seas_cooling_end = weights_seasonal_cooling[1:].repeat(current_batch_size)

        # Stack channels: 0=Normal, 1=Faded, 2=SeasonalHeating, 3=SeasonalCooling
        w_start = torch.stack([w_norm_start, w_faded_start, w_seas_heating_start, w_seas_cooling_start], dim=1)
        w_end = torch.stack([w_norm_end, w_faded_end, w_seas_heating_end, w_seas_cooling_end], dim=1)

        interpolated_weights = w_start[segment_indices] + (
            w_end[segment_indices] - w_start[segment_indices]
        ) * t_factor.unsqueeze(1)

        # Generate Batch IDs to separate lines later
        batch_ids_base = torch.arange(current_batch_size, device=device).repeat_interleave(num_segments)
        pixel_batch_ids = batch_ids_base[segment_indices]

        # --- 5. Bilinear Splatting ---
        # Distribute value to 4 integer neighbors
        x_float = interpolated_coords[:, 0]
        y_float = interpolated_coords[:, 1]
        x_floor = x_float.floor().long()
        y_floor = y_float.floor().long()

        # Neighbors: TL, TR, BL, BR
        x_neighbors = torch.stack([x_floor, x_floor + 1, x_floor, x_floor + 1], dim=1).view(-1)
        y_neighbors = torch.stack([y_floor, y_floor, y_floor + 1, y_floor + 1], dim=1).view(-1)

        # Bounds check
        valid_mask = (
            (x_neighbors >= 0) & (x_neighbors < canvas_width) & (y_neighbors >= 0) & (y_neighbors < canvas_height)
        )

        if not valid_mask.any():
            continue

        x_final = x_neighbors[valid_mask]
        y_final = y_neighbors[valid_mask]

        # Expand values and batch IDs to 4 neighbors
        values_expanded = interpolated_weights.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 4)
        values_final = values_expanded[valid_mask]

        batch_ids_expanded = pixel_batch_ids.unsqueeze(1).expand(-1, 4).reshape(-1)
        batch_ids_final = batch_ids_expanded[valid_mask]

        # --- 6. Sparse Reduction ---
        # Intra-line Max: If a line overlaps itself, take MAX to avoid hot spots.
        spatial_indices = (y_final * canvas_width) + x_final
        # Unique Hash = (BatchID * SpatialDim) + SpatialIndex
        unique_pixel_hash = (batch_ids_final * spatial_dim) + spatial_indices

        # Sort to group same-line overlaps
        sorted_indices = torch.argsort(unique_pixel_hash)
        unique_pixel_hash = unique_pixel_hash[sorted_indices]
        values_final = values_final[sorted_indices]

        # Identify unique (Batch, Pixel) pairs
        unique_hashes, inverse_indices = torch.unique_consecutive(unique_pixel_hash, return_inverse=True)

        # Reduce Step 1: Maximize within the same line
        reduced_values = torch.zeros(
            (unique_hashes.shape[0], 4),
            device=device,
            dtype=density_accumulator.dtype,
        )
        reduced_values.scatter_reduce_(
            0,
            inverse_indices.unsqueeze(1).expand(-1, 4),
            values_final,
            reduce="amax",
            include_self=False,
        )

        # Reduce Step 2: Sum/Max across different lines
        global_spatial_indices = unique_hashes % spatial_dim

        # Accumulate Sums (Density)
        total_density.view(-1).scatter_add_(0, global_spatial_indices, reduced_values[:, 0])
        total_density_sq.view(-1).scatter_add_(0, global_spatial_indices, reduced_values[:, 0].pow(2))
        total_faded.view(-1).scatter_add_(0, global_spatial_indices, reduced_values[:, 1])
        total_seasonal_heating.view(-1).scatter_add_(0, global_spatial_indices, reduced_values[:, 2])
        total_seasonal_cooling.view(-1).scatter_add_(0, global_spatial_indices, reduced_values[:, 3])

        # Accumulate Maxima
        time_max_accumulator.view(-1).scatter_reduce_(
            0,
            global_spatial_indices,
            reduced_values[:, 1],
            reduce="amax",
            include_self=True,
        )
        seasonal_max_accumulator.view(-1).scatter_reduce_(
            0,
            global_spatial_indices,
            reduced_values[:, 2],
            reduce="amax",
            include_self=True,
        )

        # Explicit cleanup
        del (
            points_chunk,
            points_centered,
            flat_points,
            segment_start,
            segment_end,
            delta,
            steps,
            segment_boundaries,
            global_pixel_indices,
            segment_indices,
            t_factor,
            interpolated_coords,
            interpolated_weights,
            x_neighbors,
            y_neighbors,
            unique_pixel_hash,
            sorted_indices,
            unique_hashes,
            inverse_indices,
            reduced_values,
        )

    # --- Final Merge ---
    # Average the summed densities over the number of samples
    total_density /= num_samples
    total_density_sq /= num_samples
    total_faded /= num_samples
    total_seasonal_heating /= num_samples
    total_seasonal_cooling /= num_samples

    variance = total_density_sq - total_density.pow(2)
    variance.clamp_(min=0)

    # 1. Calculate Absolute Uncertainty (Standard Deviation)
    # This shows the absolute spread of the streamlines at every pixel.
    uncertainty_absolute = torch.sqrt(variance)

    # 2. Calculate Relative Uncertainty (Coefficient of Variation)
    # This highlights areas where the path is "fuzzy" relative to the flow density.
    # We add epsilon to avoid division by zero in empty areas.
    epsilon = 1e-6
    _uncertainty_relative = uncertainty_absolute / (total_density + epsilon)

    # Update global accumulators using MAX blending
    density_accumulator.view(-1).copy_(torch.maximum(density_accumulator.view(-1), total_density.view(-1)))
    faded_density_accumulator.view(-1).copy_(torch.maximum(faded_density_accumulator.view(-1), total_faded.view(-1)))
    seasonal_density_accumulator_heating.view(-1).copy_(
        torch.maximum(seasonal_density_accumulator_heating.view(-1), total_seasonal_heating.view(-1))
    )
    seasonal_density_accumulator_cooling.view(-1).copy_(
        torch.minimum(seasonal_density_accumulator_cooling.view(-1), total_seasonal_cooling.view(-1))
    )
    variance_accumulator.view(-1).copy_(torch.maximum(variance_accumulator.view(-1), uncertainty_absolute.view(-1)))


def normalize_heatmap(tensor: torch.Tensor) -> torch.Tensor:
    """Normalizes a tensor to [0, 1] range, handling zero-division."""
    max_val = tensor.max()
    if max_val > 0:
        return tensor / max_val
    return tensor


@torch.jit.script
def sample_velocity(
    positions: torch.Tensor, field_tensor: torch.Tensor, inv_w: float, inv_h: float, resolution: float
) -> torch.Tensor:
    """
    Samples velocity vectors from a vector field at specific positions using bilinear interpolation.

    Args:
        positions: Coordinates tensor of shape (N, 2) where (x, y).
        field_tensor: Velocity field tensor of shape (1, 2, H, W).
        inv_w: Inverse width factor (2 / (W - 1)).
        inv_h: Inverse height factor (2 / (H - 1)).
        resolution: Scaling factor for the output velocity.

    Returns:
        Sampled velocity vectors of shape (N, 2).
    """
    N = positions.size(0)

    # 1. Normalize coordinates to [-1, 1] range for grid_sample
    nx = positions[:, 0] * inv_w - 1.0
    ny = positions[:, 1] * inv_h - 1.0

    # 2. Prepare grid for sampling
    grid = torch.stack((nx, ny), dim=-1).view(1, N, 1, 2)

    # 3. Sample
    v_sample = F.grid_sample(field_tensor, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

    # 4. Rearrange to (N, 2)
    return v_sample.view(2, N).t() / resolution


@torch.jit.script
def step_chunk_jit(
    positions: torch.Tensor,
    field_tensor: torch.Tensor,
    dt: float,
    sqrt_dt: float,
    chunk_len: int,
    width: int,
    height: int,
    resolution: float,
    diffusion_base: float,
    diffusion_scale: float,
) -> torch.Tensor:
    """
    JIT-compiled RK4 integrator with stochastic noise for streamline simulation.

    Args:
      positions: Start positions (N, 2).
      field_tensor: Velocity field (1, 2, H, W).
      dt: Time step delta.
      sqrt_dt: Square root of time step.
      chunk_len: Number of integration steps to perform.
      width: Field width (pixels).
      height: Field height (pixels).
      resolution: Grid resolution scaling factor.
      diffusion_base: Base noise amplitude.
      diffusion_scale: Velocity-dependent noise scaling.

    Returns:
      History tensor of shape (N, chunk_len, 2).
    """
    # --- Setup ---
    inv_w = 2.0 / (float(width) - 1.0)
    inv_h = 2.0 / (float(height) - 1.0)

    N = positions.size(0)
    device = positions.device
    dtype = positions.dtype

    # Pre-allocate history
    history = torch.zeros((N, chunk_len, 2), device=device, dtype=dtype)

    # Pre-calculate Gaussian noise block for Brownian motion
    noise_block = torch.randn((chunk_len, N, 2), device=device, dtype=dtype) * sqrt_dt

    # Set initial state
    history[:, 0, :] = positions
    current_pos = positions.clone()

    # --- Integration Loop ---
    for i in range(1, chunk_len):
        # 1. Deterministic Advection (Runge-Kutta 4)
        k1 = sample_velocity(current_pos, field_tensor, inv_w, inv_h, resolution)
        k2 = sample_velocity(current_pos + 0.5 * dt * k1, field_tensor, inv_w, inv_h, resolution)
        k3 = sample_velocity(current_pos + 0.5 * dt * k2, field_tensor, inv_w, inv_h, resolution)
        k4 = sample_velocity(current_pos + dt * k3, field_tensor, inv_w, inv_h, resolution)

        v_advection = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

        # 2. Stochastic Update (Dispersion)
        # Noise magnitude scales with velocity magnitude (high speed = high turbulent dispersion)
        v_mag = torch.norm(k1, dim=1, keepdim=True)
        noise_magnitude = v_mag * diffusion_scale + diffusion_base
        displacement = noise_block[i] * noise_magnitude

        # 3. Update Position
        current_pos = current_pos + v_advection * dt + displacement

        # 4. Clamp to Borders
        current_pos[:, 0] = torch.clamp(current_pos[:, 0], 0.0, width)
        current_pos[:, 1] = torch.clamp(current_pos[:, 1], 0.0, height)

        history[:, i, :] = current_pos
    return history


def make_streamlines_gpu(
    streamline_config: Any,
    physical_parameters: Any,
    heat_pump_positions: torch.Tensor,
    vx: torch.Tensor,
    vy: torch.Tensor,
    dims: tuple[int, int],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Main driver to generate stochastic streamlines on GPU.

    Args:
      mode: Simulation configuration object.
      heat_pump_positions: Starting coordinates (N, 2).
      vx: Velocity X field.
      vy: Velocity Y field.
      dims: (Width, Height).

    Returns:
      tuple of (Mean, Variance, MeanTime, MaxTime, SeasonalMax, SeasonalPos) heatmaps.
      All returned tensors are CPU resident and transposed (H, W).
    """
    if streamline_config.samples <= 0:
        raise ValueError("Samples must be greater than 0")

    device = heat_pump_positions.device
    vram_available = torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)

    info(f"Running simulation on: {device}")
    info(f"Available VRAM: {vram_available / (1024**3):.2f} GB")

    # Heuristic for batch sizing based on VRAM
    hp_batch_size = max(1, int(vram_available / (4 * 1024**3)))
    info(f"HP_BATCH_SIZE: {hp_batch_size}")

    dt = physical_parameters.duration_years / streamline_config.steps
    sqrt_dt = dt**0.5

    grid_width, grid_height = dims

    with torch.inference_mode():
        # Prepare shape for grid_sample: (Batch=1, Channels=2, H, W)
        velocity_field = torch.stack([vy, vx], dim=0).unsqueeze(0).to(device)
        velocity_height, velocity_width = vx.shape

        # Initialize Global Accumulators
        density_acc = torch.zeros((grid_height, grid_width), device=device, dtype=torch.float32)
        faded_density_acc = torch.zeros((grid_height, grid_width), device=device, dtype=torch.float32)
        time_max_acc = torch.zeros((grid_height, grid_width), device=device, dtype=torch.float32)
        seasonal_max_acc = torch.zeros((grid_height, grid_width), device=device, dtype=torch.float32)
        seasonal_density_acc_heating = torch.zeros((grid_height, grid_width), device=device, dtype=torch.float32)
        seasonal_density_acc_cooling = torch.zeros((grid_height, grid_width), device=device, dtype=torch.float32)
        variance_accumulator = torch.zeros((grid_height, grid_width), device=device, dtype=torch.float32)

        num_hps = len(heat_pump_positions)

        # Precompute weighting arrays
        steps_indices = torch.arange(0, streamline_config.steps, device=device, dtype=torch.float32)

        # normal
        weights_normal_global = torch.ones_like(steps_indices)

        # faded
        weights_faded_global = 1.0 - (steps_indices / streamline_config.steps)
        weights_faded_global = torch.clamp(weights_faded_global, 0.0, 1.0)

        # seasonal
        weights_seasonal_global = 1.0 - (steps_indices / streamline_config.steps)
        weights_seasonal_global = torch.clamp(weights_seasonal_global, 0.0, 1.0)
        (injectionRate, seasonalCycleSteps1) = convert_injection_config(
            physical_parameters.injection_rate_m3_per_s,
            streamline_config.steps,
            physical_parameters.duration_years,
            device,
        )
        (injectionTemp, seasonalCycleSteps2) = convert_injection_config(
            physical_parameters.injection_temperature_C,
            streamline_config.steps,
            physical_parameters.duration_years,
            device,
        )
        assert seasonalCycleSteps1 == seasonalCycleSteps2, (
            "Mismatch in seasonal cycle steps between rate and temperature"
        )

        injectionTemp = injectionTemp - physical_parameters.ambient_temperature_C
        injectionTemp = injectionTemp / max(abs(injectionTemp.max()), abs(injectionTemp.min()))

        weights_seasonal_global_heating = (
            weights_seasonal_global
            * (injectionRate / injectionRate.max())
            * (torch.where(injectionTemp > 0, injectionTemp, torch.zeros_like(injectionTemp)))
        )
        weights_seasonal_global_cooling = (
            weights_seasonal_global
            * (injectionRate / injectionRate.max())
            * (torch.where(injectionTemp < 0, injectionTemp, torch.zeros_like(injectionTemp)))
        )

        # Iterate through batches of Heat Pumps
        for b_idx in tqdm(
            range(0, num_hps, hp_batch_size),
            desc="Generating Streamlines",
            unit="HP Batches",
        ):
            b_end = min(b_idx + hp_batch_size, num_hps)
            batch_positions = heat_pump_positions[b_idx:b_end]

            # Expand points for stochastic sampling
            current_positions = batch_positions.repeat_interleave(streamline_config.samples, dim=0)

            # Physics Integration
            position_history = step_chunk_jit(
                current_positions,
                velocity_field,
                float(dt),
                float(sqrt_dt),
                streamline_config.steps,
                velocity_width,
                velocity_height,
                float(physical_parameters.resolution_m),
                float(streamline_config.diffusion_base),
                float(streamline_config.diffusion_scale),
            )

            del current_positions

            # Reshape to (HP_Batch, Samples, Steps, 2) for rendering
            current_batch_count = batch_positions.shape[0]
            hp_history_slices = position_history.view(
                current_batch_count, streamline_config.samples, streamline_config.steps, 2
            )

            # Rasterization
            for i in range(current_batch_count):
                draw_batched_polylines(
                    density_acc,
                    faded_density_acc,
                    time_max_acc,
                    seasonal_max_acc,
                    seasonal_density_acc_heating,
                    seasonal_density_acc_cooling,
                    variance_accumulator,
                    hp_history_slices[i],
                    weights_normal_global,
                    weights_faded_global,
                    weights_seasonal_global_heating,
                    weights_seasonal_global_cooling,
                    dims,
                )

            del position_history

        # --- Post-Processing ---
        # Normalize variance, all others should already be normed
        variance_norm = normalize_heatmap(variance_accumulator)

        seasonal_density_acc = seasonal_density_acc_heating + seasonal_density_acc_cooling
        if 0 <= seasonal_density_acc.min().item():
            # rescale seasonal density from [0, 1] to [0, 1]
            seasonal_density_acc = seasonal_density_acc
        elif seasonal_density_acc.max().item() <= 0:
            # rescale seasonal density from [-1, 0] to [0, 1]
            seasonal_density_acc = seasonal_density_acc + 1
        else:
            # rescale seasonal density from [-1, 1] to [0, 1]
            seasonal_density_acc = (seasonal_density_acc + 1) / 2

        # Validation Checks
        check_list = {
            "density_acc": density_acc,
            "variance_norm": variance_norm,
            "faded_density_acc": faded_density_acc,
            "time_max_acc": time_max_acc,
            "seasonal_max_acc": seasonal_max_acc,
            "seasonal_density_acc": seasonal_density_acc,
        }
        for key, vec in check_list.items():
            if (vec < 0).any():
                raise ValueError(f"Output map {key} contains Negative values")
            if torch.isnan(vec).any():
                raise ValueError(f"Output map {key} contains NaN values")
            if torch.isinf(vec).any():
                raise ValueError(f"Output map {key} contains Inf values")

        return (
            density_acc.t().cpu(),
            variance_norm.t().cpu(),
            faded_density_acc.t().cpu(),
            time_max_acc.t().cpu(),
            seasonal_max_acc.t().cpu(),
            seasonal_density_acc.t().cpu(),
        )
