"""Shared helpers for pipeline smoke tests."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent


def run_pipeline(config) -> None:
    """Execute AppConfig.pipeline the same way as ``python -m code``."""
    from code.__main__ import clean_target, set_seed
    from code.processing.cnn_main import step_cnn
    from code.streamlines.streamline_main import execute_streamline_pipeline
    from code.utils.utils_args import save_yaml

    set_seed(config.run_configuration.seed)
    run_dir = config.paths.results / config.run_configuration.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(config.model_dump(), run_dir / "config.yaml")

    for action in config.run_configuration.pipeline:
        for key, value in action.items():
            match key:
                case "clean":
                    clean_target(config, value)
                case "step1":
                    step_cnn(
                        config.run_configuration, config.paths, config.general_configuration.step1, "step1", value
                    )
                case "step2":
                    execute_streamline_pipeline(config, value)
                case "step3":
                    step_cnn(
                        config.run_configuration, config.paths, config.general_configuration.step3, "step3", value
                    )
