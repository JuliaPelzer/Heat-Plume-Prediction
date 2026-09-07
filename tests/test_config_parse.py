"""Config parse smoke tests."""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestConfigParse(unittest.TestCase):
    def test_parse_template_yaml(self):
        from code.utils.yaml_parser import parse_config

        config = parse_config(str(ROOT / "settings" / "template.yaml"))
        self.assertEqual(config.run_configuration.run_name, "template_name")
        self.assertTrue(any("step2" in action for action in config.run_configuration.pipeline))
        self.assertEqual(config.general_configuration.step2.physical_parameters.duration_years, 27.5)


if __name__ == "__main__":
    unittest.main()
