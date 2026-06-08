import enum
import os
import pathlib
from code.preprocessing.transforms import NormalizeTransform
from code.utils import logging as log  # noqa: F401
from code.utils.utils_args import get_run_ids_from_prep

import torch
import yaml
from torch.utils.data import Dataset


class DatasetType(enum.Enum):
    unknown = 0
    seasonal = 1
    steady_state_cooling = 2
    steady_state_heating = 3


class DatasetBasis(Dataset):
    def __init__(self, path: str, box_size: int = None, idx: int = None):
        Dataset.__init__(self)
        self.path = pathlib.Path(path)
        self.info = self.__load_info()
        self.norm = NormalizeTransform(self.info)
        self.input_names = []
        self.label_names = []
        for filename in os.listdir(self.path / "Inputs"):
            self.input_names.append(filename)
        for filename in os.listdir(self.path / "Labels"):
            self.label_names.append(filename)
        self.input_names.sort()
        self.label_names.sort()
        self.spatial_size = torch.load(self.path / "Labels" / self.label_names[0]).shape[1:]
        if box_size is not None:
            self.box_size = box_size
        else:
            self.box_size = self.spatial_size[0]

        if len(self.input_names) != len(self.label_names):
            raise ValueError("Number of Inputs and labels does not match!")

    @property
    def input_channels(self):
        if hasattr(self, "_input_channels_override") and self._input_channels_override is not None:
            return self._input_channels_override
        return len(self.info["Inputs"])

    @property
    def output_channels(self):
        return len(self.info["Labels"])

    def __load_info(self):
        with open(self.path / "info.yaml") as f:
            info = yaml.safe_load(f)
        return info

    def __len__(self):
        return len(self.input_names)

    def __getitem__(self, idx):
        input = torch.load(self.path / "Inputs" / self.input_names[idx])[:, : self.box_size, :]
        label = torch.load(self.path / "Labels" / self.label_names[idx])[:, : self.box_size, :]
        return input, label


class DataPoint(DatasetBasis):
    def __init__(self, path: str, i: int = 0):
        DatasetBasis.__init__(self, path)
        if isinstance(i, int):
            run_id = get_run_ids_from_prep(self.path / "Inputs")[i]

            self.input_names = [f"RUN_{run_id}.pt"]
            self.label_names = [f"RUN_{run_id}.pt"]
        elif isinstance(i, list):
            self.input_names = [f"RUN_{get_run_ids_from_prep(self.path / 'Inputs')[ii]}.pt" for ii in i]
            self.label_names = [f"RUN_{get_run_ids_from_prep(self.path / 'Labels')[ii]}.pt" for ii in i]
        else:
            raise ValueError("i must be an int or a list of ints")
        self.input_names.sort()
        self.label_names.sort()


class DataPointSequence(DataPoint):
    def __init__(self, path: str, i: int = 0, time_steps_to_predict=None, max_simulation_timestep=None):
        DataPoint.__init__(self, path, i)
        self.time_steps_to_predict = time_steps_to_predict
        self.max_simulation_timestep = max_simulation_timestep

        if isinstance(time_steps_to_predict, list) and len(time_steps_to_predict) > 0:
            if isinstance(time_steps_to_predict[0], list):
                self.sublist_mode = True
                self.subsets = time_steps_to_predict
                lengths = [len(sub) for sub in self.subsets]
                if len(set(lengths)) != 1:
                    raise ValueError("All time_steps_to_predict sublists must have the same length")
                self.output_length = lengths[0]
            else:
                self.sublist_mode = False
                self.subsets = [time_steps_to_predict]
                self.output_length = len(time_steps_to_predict)
        else:
            self.sublist_mode = False
            self.subsets = []
            self.output_length = None

        if self.max_simulation_timestep is None:
            # infer from labels available in first sample
            sample_labels = torch.load(self.path / "Labels" / self.label_names[0])
            if sample_labels.dim() >= 2:
                self.max_simulation_timestep = sample_labels.shape[1] - 1

    def __len__(self):
        if self.sublist_mode and len(self.subsets) > 0:
            return len(self.input_names) * len(self.subsets)
        return len(self.input_names)

    def __getitem__(self, idx):
        if self.sublist_mode and len(self.subsets) > 0:
            run_idx = idx // len(self.subsets)
            subset_idx = idx % len(self.subsets)
            sublist = self.subsets[subset_idx]
        else:
            run_idx = idx
            subset_idx = 0
            sublist = self.subsets[0] if self.subsets else None

        inputs = torch.load(self.path / "Inputs" / self.input_names[run_idx])
        labels = torch.load(self.path / "Labels" / self.label_names[run_idx])

        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(1)

        if sublist is not None:
            #        # ── Select labels for this sublist ──────────────────────────────
            # if self.sublist_mode:
            #     flat = [t for sub in self.subsets for t in sub]
            #     min_time = min(flat)
            #     label_indices = [t - min_time for t in sublist]
            # else:
            #     label_indices = list(range(len(sublist)))
            labels = labels[:, sublist, :, :]

            # ── Base input frame ────────────────────────────────────────────
            input_timestep = min(max(0, sublist[0] - 1), inputs.shape[1] - 1)
            base_input = inputs[:, input_timestep, :, :].unsqueeze(1)
            seq_len = len(sublist)
            base_input = base_input.repeat(1, seq_len, 1, 1)  # (C, seq_len, H, W)

            # ── Time-encoding channels (must match cuts dataset) ────────────
            H, W = inputs.shape[-2], inputs.shape[-1]
            max_t = self.max_simulation_timestep
            abs_vals = torch.tensor([t / max_t for t in sublist], dtype=inputs.dtype)
            abs_channel = abs_vals.view(1, seq_len, 1, 1).expand(1, seq_len, H, W)
            gap_vals = torch.tensor([(t - input_timestep) / max_t for t in sublist], dtype=inputs.dtype)
            gap_channel = gap_vals.view(1, seq_len, 1, 1).expand(1, seq_len, H, W)

            inputs = torch.cat((base_input, abs_channel, gap_channel), dim=0)

        # Return metadata for chained iterative training
        metadata = {
            "run_idx": run_idx,
            "chain_idx": run_idx,  # ← was missing; run_idx is the stable chain id here
            "subset_idx": subset_idx,
            "sublist_mode": self.sublist_mode,
            "max_simulation_timestep": self.max_simulation_timestep,
            "pos": (0, 0),  # ← no spatial crop, but key must exist if anything reads it
            "sublist": sublist if sublist is not None else [],
        }
        return inputs, labels, metadata
