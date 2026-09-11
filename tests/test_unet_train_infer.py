"""U-Net train + infer smoke test."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader, TensorDataset

SMOKE_UNET_OUT = Path(__file__).resolve().parent / "heatplume_unet_smoke"


class TestUnetTrainInfer(unittest.TestCase):
    def test_unet_train_then_infer_smoke(self):
        """One-epoch U-Net train via Solver, then save/load + Model.infer.

        Writes to tests/heatplume_unet_smoke/; not cleaned up.
        """
        from code.preprocessing.datasets.dataset import DatasetType
        from code.processing.networks.unetVariants import UNet
        from code.processing.solver import Solver

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(0)
        h = w = 32
        batch, cin, cout, depth = 2, 3, 2, 2
        n = 4

        x = torch.randn(n, cin, h, w)
        y = torch.randn(n, cout, h, w)
        ds = TensorDataset(x, y)
        loader = DataLoader(ds, batch_size=batch, shuffle=False, drop_last=False, num_workers=0)

        model = UNet(cin, cout, init_features=8, depth=depth, kernel_size=3, norm="batchnorm").float().to(device)
        solver = Solver(model, ds, ds, loss_func=MSELoss(), batchsize=batch, finetune=False)

        dest = SMOKE_UNET_OUT
        dest.mkdir(parents=True, exist_ok=True)
        args = {
            "destination": dest,
            "device": device,
            "epochs": 1,
            "scheduler": SimpleNamespace(
                type="ReduceLROnPlateau",
                init_lr=1e-3,
                mode="min",
                factor=0.5,
                patience=2,
                threshold=1e-3,
                min_lr=1e-6,
            ),
        }
        # Solver.train always visualizes; skip I/O that needs real DataPoint norms.
        with patch("code.processing.solver.visualize_outputs"):
            val_loss = solver.train(DatasetType.unknown, loader, loader, args)

        self.assertTrue(torch.isfinite(torch.tensor(val_loss)))
        model.save(dest)
        self.assertTrue((dest / "model.pt").is_file())

        loaded = UNet(cin, cout, init_features=8, depth=depth, kernel_size=3, norm="batchnorm").float()
        loaded.load(dest, device)
        pred = loaded.infer(x[:1], device)
        torch.save(pred.cpu(), dest / "infer_pred.pt")

        self.assertEqual(tuple(pred.shape), (1, cout, h, w))
        self.assertTrue(torch.isfinite(pred).all())


if __name__ == "__main__":
    unittest.main()
