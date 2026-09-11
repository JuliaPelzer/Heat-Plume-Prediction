from code.preprocessing.datasets.dataset import DatasetBasis
from code.utils import logging as log  # noqa: F401
from code.utils.utils_args import get_run_ids_from_prep
from pathlib import Path

import numpy as np
import torch


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
        self.inputs = torch.stack(self.inputs)  # (datapoints, channels, time, H, W)
        self.labels = torch.stack(self.labels)
        # switch dim D and channels
        self.inputs = self.inputs.permute(1, 0, 2, 3, 4)
        self.labels = self.labels.permute(1, 0, 2, 3, 4)
        assert len(self.inputs.shape) == 5, "inputs should be 5D (channels, datapoints, time, H,W)"
        assert self.inputs.shape[-2:] == self.labels.shape[-2:], "inputs and labels should have same shape"

        self.spatial_size = self.inputs.shape[-2:]
        self.n_dp = self.inputs.shape[1]
        if not self.inputs.shape[-2:] == self.labels.shape[-2:]:
            raise ValueError(
                f"Spatial size of inputs {self.inputs.shape[-2:]} and labels {self.labels.shape[-2:]} do not match. See SimulationDatasetCut to fix."
            )

        self.box_size = np.array([box_size, box_size])
        self.skip_per_dir = skip_per_dir
        self.case = case

    def __getitem__(self, i):
        id, pos = self.idx_to_pos(i)
        # c  id  t  h                               w
        inputs = self.inputs[:, id, :, pos[0] : pos[0] + self.box_size[0], pos[1] : pos[1] + self.box_size[1]]
        labels = self.labels[:, id, :, pos[0] : pos[0] + self.box_size[0], pos[1] : pos[1] + self.box_size[1]]
        inputs = torch.cat((inputs, labels[:, 0, :, :].unsqueeze(1)), dim=0)
        labels = labels[:, 1:, :, :]
        return inputs, labels
