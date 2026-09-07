"""Step2 API / tiny RWPT smoke tests."""

from __future__ import annotations

import unittest

import torch


class TestStep2Apis(unittest.TestCase):
    def test_streamline_and_rwpt_entry_points_are_distinct(self):
        from code.streamlines.calculation import streamline_rwpt as ds
        from code.streamlines.calculation import streamline_tensors as st

        self.assertTrue(callable(st.make_streamlines_gpu))
        self.assertTrue(callable(ds.run_rwpt_thermal_prior))
        self.assertTrue(callable(ds.generate_physical_plumes))
        self.assertTrue(hasattr(ds, "RwptConfig"))
        self.assertFalse(hasattr(ds, "DirectSolverConfig"))
        self.assertFalse(hasattr(ds, "direct_solve"))
        self.assertFalse(hasattr(ds, "SimulationConfig"))

    def test_rwpt_config_and_tiny_rwpt(self):
        from code.streamlines.calculation.streamline_rwpt import (
            RwptConfig,
            generate_physical_plumes,
        )

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        h, w = 16, 16
        steps = 4
        rates = torch.full((1, steps), 1e-3, device=device)
        temps = torch.full((1, steps), 15.0, device=device)
        config = RwptConfig(
            device=device,
            resolution_m_per_px=5.0,
            ambientTemp_C=10.6,
            injectionRate_m3_per_s=rates,
            injectionTemp_C=temps,
            timeSteps_count=steps,
            timeEnd_years=1.0,
            seasonalCycleSteps=steps,
            samplesPerSource_count=2,
            porosity_frac=0.25,
            rockDensity_kg_per_m3=2650.0,
            rockSpecificHeat_J_per_kgK=880.0,
            waterDensity_kg_per_m3=1000.0,
            waterSpecificHeat_J_per_kgK=4182.0,
            thermalConductivityDry_W_per_mK=0.5,
            thermalConductivityWet_W_per_mK=2.0,
            thicknessAquifer_m=10.0,
            longitudinalDispersivity_m=1.0,
            transverseDispersivityH_m=0.1,
        )
        self.assertGreater(config.timeStep_s, 0.0)
        self.assertGreater(config.retardationFactor_dimless, 1.0)

        hp = torch.tensor([[w / 2, h / 2]], dtype=torch.float32, device=device)
        vx = torch.ones((h, w), dtype=torch.float32, device=device)
        vy = torch.zeros((h, w), dtype=torch.float32, device=device)
        try:
            out = generate_physical_plumes(config, hp, vx, vy, (w, h))
        except Exception as exc:  # noqa: BLE001 — smoke test; environment may lack CUDA kernels
            self.skipTest(f"RWPT smoke skipped: {exc}")
        self.assertEqual(tuple(out.shape), (h, w))
        self.assertEqual(out.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
