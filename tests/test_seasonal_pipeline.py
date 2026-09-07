"""Seasonal pipeline smoke test (settings/seasonal-test.yaml)."""

from __future__ import annotations

import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from helpers import ROOT, run_pipeline

SMOKE_SEASONAL_OUT = Path(__file__).resolve().parent / "heatplume_seasonal_smoke"


def _ensure_minimal_seasonal_raw(smoke_root: Path) -> Path:
    """Symlink two RUN folders + settings into an isolated smoke dataset."""
    src = ROOT / "datasets" / "paper-results-seasonal"
    if not (src / "RUN_0" / "pflotran.h5").is_file():
        raise unittest.SkipTest(f"Seasonal raw data missing under {src}")

    dst = smoke_root / "datasets" / "paper-results-seasonal"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("RUN_0", "RUN_1", "settings.yaml", "inputs"):
        link = dst / name
        if link.exists() or link.is_symlink():
            continue
        link.symlink_to(src / name)
    return dst


class TestSeasonalPipelineSmoke(unittest.TestCase):
    def test_seasonal_yaml_minimal_pipeline(self):
        """Run settings/seasonal-test.yaml (1 epoch, 2 RUNs, tiny streamlines).

        Artifacts under tests/heatplume_seasonal_smoke/ (not cleaned up).
        Full step1→2→3 runs on CPU with the smoke settings (1000², few HPs).
        """
        from code.utils.yaml_parser import parse_config

        if not (ROOT / "datasets" / "paper-results-seasonal" / "RUN_0" / "pflotran.h5").is_file():
            self.skipTest("paper-results-seasonal dataset not available")

        smoke_root = SMOKE_SEASONAL_OUT
        smoke_root.mkdir(parents=True, exist_ok=True)
        _ensure_minimal_seasonal_raw(smoke_root)

        config = parse_config(str(ROOT / "settings" / "seasonal-test.yaml"))
        if not torch.cuda.is_available():
            config.run_configuration.device = "cpu"

        def _cpu_safe_dataloader(batchsize: int, dataset, shuffle: bool = True):
            return DataLoader(
                dataset,
                batch_size=min(len(dataset), batchsize),
                shuffle=shuffle,
                drop_last=False,
                num_workers=0,
            )

        with ExitStack() as stack:
            stack.enter_context(patch("code.processing.solver.visualize_outputs"))
            stack.enter_context(patch("code.streamlines.streamline_main.run_visualization"))
            stack.enter_context(
                patch("code.processing.training.construct_dataloader", side_effect=_cpu_safe_dataloader)
            )
            if str(config.run_configuration.device) == "cpu" or not torch.cuda.is_available():
                stack.enter_context(
                    patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(total_memory=8 * 1024**3))
                )
                stack.enter_context(patch("torch.cuda.memory_allocated", return_value=0))
                stack.enter_context(patch("torch.cuda.synchronize"))
            try:
                run_pipeline(config)
            except Exception as exc:  # noqa: BLE001
                self.fail(f"seasonal smoke pipeline failed: {exc}")

        run_dir = Path(config.paths.results) / config.run_configuration.run_name
        self.assertTrue((run_dir / "step1" / "model.pt").is_file())
        self.assertTrue((run_dir / "config.yaml").is_file())
        self.assertTrue((run_dir / "step3" / "model.pt").is_file())


if __name__ == "__main__":
    unittest.main()
