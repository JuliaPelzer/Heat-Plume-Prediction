import math
import os
import random
from code.preprocessing.datasets.dataset import DatasetBasis
from code.utils import logging as log  # noqa: F401
from code.utils.utils_args import get_run_ids_from_prep
from code.utils.utils_args import log as log
from pathlib import Path

import numpy as np
import torch
from attr import dataclass


class SimulationDatasetCuts(DatasetBasis):
    def __init__(
        self,
        path: Path,
        skip_per_dir: int = 4,
        box_size: int = 64,
        ids: list[int] | None = None,
        case: str = "train",
    ):
        if ids is None:
            ids = [0]
        DatasetBasis.__init__(self, path, box_size)

        if isinstance(ids, int):  # handle ids = list AND int
            ids = [ids]

        run_ids_all = get_run_ids_from_prep(self.path / "Inputs")
        run_ids = [run_ids_all[id] for id in ids]
        self.inputs = []
        self.labels = []
        for run_id in run_ids:
            self.inputs.append(torch.load(self.path / "Inputs" / f"RUN_{run_id}.pt"))
            self.labels.append(torch.load(self.path / "Labels" / f"RUN_{run_id}.pt"))
        self.inputs = torch.stack(self.inputs)
        self.labels = torch.stack(self.labels)
        # switch dim D and channels
        self.inputs = self.inputs.permute(1, 0, 2, 3)
        self.labels = self.labels.permute(1, 0, 2, 3)
        assert len(self.inputs.shape) == 4, (
            "inputs should be 4D (C,D,H,W), D = datapoints, C = channels, H = height, W = width"
        )

        self.spatial_size = self.inputs.shape[2:]
        self.n_dp = self.inputs.shape[1]
        if not self.inputs.shape[2:] == self.labels.shape[2:]:
            required_shape = self.spatial_size
            start_pos = [
                self.labels.shape[2] // 2 - required_shape[0] // 2,
                self.labels.shape[3] // 2 - required_shape[1] // 2,
            ]
            self.labels = self.labels[
                :, :, start_pos[0] : start_pos[0] + required_shape[0], start_pos[1] : start_pos[1] + required_shape[1]
            ]

        assert self.inputs.shape[1:] == self.labels.shape[1:], "inputs and labels should have same shape"

        self.box_size = np.array([box_size, box_size])
        self.box_out = np.array([0, 0]).astype(int)
        self.skip_per_dir = skip_per_dir
        self.case = case

    def __len__(self):
        return (
            self.n_dp
            * (self.spatial_size[0] - self.box_size[0])
            * (self.spatial_size[1] - self.box_size[1])
            // self.skip_per_dir**2
        )

    def __getitem__(self, i):
        id, pos = self.idx_to_pos(i)
        # assert id too close to wall
        # assert (pos+self.box_size < self.spatial_size).all(), "box too close to wall" too expensive in every call
        inputs = self.inputs[:, id, pos[0] : pos[0] + self.box_size[0], pos[1] : pos[1] + self.box_size[1]]
        labels = self.labels[
            :,
            id,
            pos[0] + self.box_out[0] : pos[0] + self.box_size[0] - self.box_out[0],
            pos[1] + self.box_out[1] : pos[1] + self.box_size[1] - self.box_out[1],
        ]
        return inputs, labels

    def idx_to_pos(self, i):
        # idx zerlegen in mod und div
        idx_mod = i % self.n_dp
        idx_div = i // self.n_dp
        return idx_mod, np.array(
            [
                ((idx_div * self.skip_per_dir) // (self.spatial_size[1] - self.box_size[1])) * self.skip_per_dir,
                (idx_div * self.skip_per_dir) % (self.spatial_size[1] - self.box_size[1]),
            ]
        )


class SimulationDatasetCutsSequential(SimulationDatasetCuts):
    def __init__(
        self,
        path: Path,
        time_steps_to_predict,
        max_simulation_timestep: int,
        skip_per_dir: int = 4,
        box_size: int = 64,
        ids: list[int] | None = None,
        case: str = "train",
        log_path: Path | None = None,
        ratio_blank_boxes: float = 0.1,
    ):
        if ids is None:
            ids = [0]
        DatasetBasis.__init__(self, path, box_size)
        if log_path is not None:
            log.configure_logging(log_path=log_path, clear_handlers=False)

        if isinstance(ids, int):  # handle ids = list AND int
            ids = [ids]

        self.max_simulation_timestep = max_simulation_timestep
        run_ids_all = get_run_ids_from_prep(self.path / "Inputs")
        run_ids = [run_ids_all[id] for id in ids]
        self.inputs = []
        self.labels = []
        for run_id in run_ids:
            self.inputs.append(torch.load(self.path / "Inputs" / f"RUN_{run_id}.pt"))
            self.labels.append(torch.load(self.path / "Labels" / f"RUN_{run_id}.pt"))
        self.inputs = torch.stack(self.inputs)  # (datapoints, channels, time, H, W)
        self.labels = torch.stack(self.labels)

        # switch dim D and channels
        if len(self.inputs.shape) == 4:
            self.inputs = self.inputs.unsqueeze(2)

        self.inputs = self.inputs.permute(1, 0, 2, 3, 4)
        self.labels = self.labels.permute(1, 0, 2, 3, 4)

        self.time_steps_to_predict = time_steps_to_predict
        if len(time_steps_to_predict) > 1:
            self.sublist_mode = True
            self.subsets = time_steps_to_predict
            self.output_length = len(time_steps_to_predict[0])
            if len(set(len(sub) for sub in self.subsets)) != 1:
                raise ValueError("All time_steps_to_predict sublists must have the same length")
        else:
            self.sublist_mode = False
            self.subsets = [time_steps_to_predict[0]] if time_steps_to_predict is not None else []
            self.output_length = len(time_steps_to_predict) if time_steps_to_predict is not None else None
        assert len(self.inputs.shape) == 5, "inputs should be 5D (channels, datapoints, time, H,W)"
        assert self.inputs.shape[-2:] == self.labels.shape[-2:], "inputs and labels should have same shape"

        # Only keep time steps to predict
        if (
            isinstance(time_steps_to_predict, list)
            and len(time_steps_to_predict) > 0
            and isinstance(time_steps_to_predict[0], list)
        ):
            flat = [t for sub in time_steps_to_predict for t in sub]
            required_time_steps = list(range(min(flat), max(flat) + 1))
        else:
            required_time_steps = list(range(min(time_steps_to_predict), max(time_steps_to_predict) + 1))
        self.labels = self.labels[:, :, required_time_steps, :, :]

        assert self.inputs.dim() == self.labels.dim() == 5, (
            f"inputs and labels should be 5D (channels, datapoints, time, H,W), but got {self.inputs.shape} and {self.labels.shape}"
        )

        self.spatial_size = self.inputs.shape[-2:]
        self.n_dp = self.inputs.shape[1]
        if not self.inputs.shape[-2:] == self.labels.shape[-2:]:
            raise ValueError(
                f"Spatial size of inputs {self.inputs.shape[-2:]} and labels {self.labels.shape[-2:]} do not match. See SimulationDatasetCut to fix."
            )

        self.box_size = np.array([box_size, box_size])
        self.skip_per_dir = skip_per_dir
        self.case = case
        self.usable_indices = None

        # Fill valid indices with indices that result in a box where the maximum material id (self.inputs[0]) is 1.
        valid_indices = []
        non_valid_indices = []
        n_h = (self.spatial_size[0] - self.box_size[0]) // self.skip_per_dir
        n_w = (self.spatial_size[1] - self.box_size[1]) // self.skip_per_dir
        total = self.n_dp * n_h * n_w

        for i in range(total):
            id, pos = self.idx_to_pos(i)
            box = self.inputs[0, id, :, pos[0] : pos[0] + self.box_size[0], pos[1] : pos[1] + self.box_size[1]]
            if box.max() == 1:
                valid_indices.append((id, pos))
            else:
                non_valid_indices.append((id, pos))

        n_blank = int(len(valid_indices) * ratio_blank_boxes)
        combined = valid_indices + non_valid_indices[:n_blank]
        self.usable_indices = combined

        # --- Extra chords: active tiles where the active pixel is at the lower border ---
        # Analogous to infer_tiled's extra_coords, but tile is positioned so the
        # active pixel falls near the bottom edge (lower 1/6) of the box.
        extra_indices = []
        H, W = self.spatial_size
        bh, bw = self.box_size  # both == box_size
        for id in range(self.n_dp):
            # active pixels: where material channel (inputs[0]) == 1.0, across all time steps
            active_mask = (self.inputs[0, id] == 1.0).any(dim=0)  # (H, W)
            active_ys, active_xs = torch.where(active_mask)

            seen = set()
            for ay, ax in zip(active_ys.tolist(), active_xs.tolist()):
                # Place tile so active pixel is near the lower border (~1/6 from bottom)
                y0 = int(np.clip(ay - bh + 5 * bh // 6, 0, H - bh))
                x0 = int(np.clip(ax - bw // 2, 0, W - bw))
                coord = (id, np.array([y0, x0]))
                key = (id, y0, x0)
                if key in seen:
                    continue
                seen.add(key)

                # Only add if the box actually contains material (active tile)
                box = self.inputs[0, id, :, y0 : y0 + bh, x0 : x0 + bw]
                if box.max() == 1.0:
                    extra_indices.append((id, np.array([y0, x0])))
                    label_box = self.labels[:, id, :, y0 : y0 + bh, x0 : x0 + bw]
                    # self._save_active_tile_label(label_box, id, y0, x0)

        self.chains = self.usable_indices + extra_indices
        random.shuffle(self.chains)

        n_valid = len(valid_indices)
        n_blank_used = min(n_blank, len(non_valid_indices))
        n_blank_discard = len(non_valid_indices) - n_blank_used
        n_extra = len(extra_indices)
        n_chains = len(self.chains)

        log.info(
            f"Dataset built | "
            f"Total grid boxes scanned: {total} | "
            f"Valid (material=1): {n_valid} ({100 * n_valid / total:.1f}%) | "
            f"Blank boxes added ({ratio_blank_boxes:.0%} of valid): {n_blank_used} | "
            f"Blank boxes discarded: {n_blank_discard} | "
            f"Extra lower-border chord tiles added: {n_extra} | "
            f"Total chains: {n_chains} | "
            f"Subsets per chain: {len(self.subsets)} | "
            f"Total dataset length: {n_chains * len(self.subsets)}"
        )

    def __len__(self):
        # if self.usable_indices is not None:
        #     return len(self.usable_indices)
        # return self.n_dp * (self.spatial_size[0] - self.box_size[0]) * (self.spatial_size[1] - self.box_size[1]) // self.skip_per_dir**2
        return len(self.chains) * len(self.subsets)

    def __getitem__(self, i):

        chain_idx = i // len(self.subsets)
        subset_idx = i % len(self.subsets)

        id, pos = self.chains[chain_idx]
        sublist = self.subsets[subset_idx]

        inputs = self.inputs[:, id, :, pos[0] : pos[0] + self.box_size[0], pos[1] : pos[1] + self.box_size[1]]
        labels = self.labels[:, id, :, pos[0] : pos[0] + self.box_size[0], pos[1] : pos[1] + self.box_size[1]]

        if self.sublist_mode:
            min_time = min(t for sub in self.subsets for t in sub)

            label_indices = [t - min_time for t in sublist]
            labels = labels[:, label_indices, :, :]

            input_timestep = min(max(0, sublist[0] - 1), inputs.shape[1] - 1)
            base_input = inputs[:, input_timestep, :, :].unsqueeze(1)
            seq_len = len(sublist)
            base_input = base_input.repeat(1, seq_len, 1, 1)

            max_t = self.max_simulation_timestep
            # rel_vals = torch.tensor([i_ / (seq_len - 1) for i_ in range(seq_len)], dtype=inputs.dtype)
            # rel = rel_vals.view(1, seq_len, 1, 1).expand(1, seq_len, *self.box_size)
            abs_vals = torch.tensor([t / max_t for t in sublist], dtype=inputs.dtype)
            abs_channel = abs_vals.view(1, seq_len, 1, 1).expand(1, seq_len, *self.box_size)
            gap_vals = torch.tensor([(t - input_timestep) / max_t for t in sublist], dtype=inputs.dtype)
            gap_channel = gap_vals.view(1, seq_len, 1, 1).expand(1, seq_len, *self.box_size)
            # inputs = torch.cat((base_input, abs_channel, gap_channel), dim=0)
            inputs = torch.cat((base_input, abs_channel, gap_channel), dim=0)

        metadata = {
            "run_idx": id,
            "chain_idx": chain_idx,  # NEW: stable identity for caching
            "subset_idx": subset_idx,
            "sublist_mode": self.sublist_mode,
            "pos": tuple(pos) if isinstance(pos, np.ndarray) else pos,
            "sublist": sublist if self.sublist_mode else [],
        }

        return inputs, labels, metadata

    def _save_active_tile_label(
        self, labels: torch.Tensor, id: int, y0: int, x0: int, debug_dir: str = "debug_tiles_dataset"
    ):
        """Save a plot of each time step in an active tile's label for confirmation."""
        import matplotlib.pyplot as plt

        os.makedirs(debug_dir, exist_ok=True)
        # labels: (channels, time, H, W) -> take first channel, sigmoid not needed (raw labels)
        frames = labels[0].cpu().numpy()  # (time, bh, bw)
        n = len(frames)
        fig, axes = plt.subplots(1, n, figsize=(3 * n, 3), squeeze=False)
        for t, ax in enumerate(axes[0]):
            im = ax.imshow(frames[t], vmin=0, vmax=1, cmap="hot")
            ax.set_title(f"t={t}")
            ax.axis("off")
        plt.colorbar(im, ax=axes[0][-1], fraction=0.046, pad=0.04)
        fig.suptitle(f"Active tile label  id={id}  y={y0}  x={x0}", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(debug_dir, f"active_tile_id{id:02d}_y{y0:04d}_x{x0:04d}.png"), dpi=100)
        plt.close(fig)


class ChainedSubsetBatchSampler(torch.utils.data.Sampler):
    """
    Yields batches where every sample shares the same subset_idx.
    Iterates subset_idx = 0, 1, 2, ... in order.
    Within each subset, chains are shuffled (if shuffle=True).

    This guarantees:
      - cache is always populated before it's read
      - no batch mixes subset indices
    """

    def __init__(self, n_chains: int, n_subsets: int, batch_size: int, shuffle: bool = True):
        self.n_chains = n_chains
        self.n_subsets = n_subsets
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        # Shuffle chain order once per epoch, same order for all subsets
        chain_order = torch.randperm(self.n_chains).tolist() if self.shuffle else list(range(self.n_chains))

        for subset_idx in range(self.n_subsets):
            # Compute flat dataset indices for this subset
            indices = [chain_idx * self.n_subsets + subset_idx for chain_idx in chain_order]
            # Yield in batches
            for start in range(0, len(indices), self.batch_size):
                yield indices[start : start + self.batch_size]

    def __len__(self):
        batches_per_subset = math.ceil(self.n_chains / self.batch_size)
        return self.n_subsets * batches_per_subset


@dataclass
class BatchContext:
    init_frame: torch.Tensor | None = None

    def update(self, y_pred):
        pass


class DefaultBatchContext(BatchContext):
    pass


class SublistBatchContext(BatchContext):
    def __init__(self):
        self.cache = {}

    def build_init_frame(self, x, metadata_list):
        subset_idx = metadata_list["subset_idx"][0].item()
        chain_ids = metadata_list["chain_idx"].tolist()

        if subset_idx == 0:
            return None

        H, W = x.shape[3], x.shape[4]

        init_frame = torch.zeros(x.shape[0], 1, H, W, device=x.device, dtype=x.dtype)

        for batch_i, chain_id in enumerate(chain_ids):
            cache_key = (chain_id, subset_idx - 1)

            if cache_key in self.cache:
                prev_pred = self.cache[cache_key].to(x.device)
                init_frame[batch_i, 0] = prev_pred[0, -1]

        return init_frame

    def update(self, metadata_list, y_pred):
        subset_idx = metadata_list["subset_idx"][0].item()
        chain_ids = metadata_list["chain_idx"].tolist()

        for batch_i, chain_id in enumerate(chain_ids):
            self.cache[(chain_id, subset_idx)] = y_pred[batch_i].detach().cpu()

        if subset_idx > 0:
            for chain_id in chain_ids:
                self.cache.pop((chain_id, subset_idx - 1), None)
