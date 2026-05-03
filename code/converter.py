import json
import os
import shutil
import sys
from code.utils import logging as log  # noqa: F401
from typing import Any

import h5py
import yaml


def build_settings_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Extracts and transforms JSON data into the target YAML structure."""
    gen = data.get("general", {})

    # Grid calculations
    ncells = gen.get("number_cells", [0, 0, 0])
    res = gen.get("cell_resolution", 5.0)

    return {
        "general": {"dimensions": 2, "random_bool": False, "seed_id": gen.get("random_seed", 0)},
        "grid": {"ncells": ncells, "size": [n * res for n in ncells]},
    }


def convert_state_to_yaml(input_dir: str, output_dir: str) -> None:
    """Reads state JSON, transforms it, and writes config YAML."""

    # copy all json/yaml files from input_dir to output_dir except
    for file_name in os.listdir(input_dir):
        if file_name == "state.json":
            json_path = os.path.join(input_dir, "state.json")
            yaml_path = os.path.join(output_dir, "inputs", "settings.yaml")

            try:
                with open(json_path) as f:
                    data = json.load(f)

                yaml_content = build_settings_dict(data)

                os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
                with open(yaml_path, "w") as f:
                    yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

            except (OSError, json.JSONDecodeError) as e:
                log.info(f"Error processing configuration files: {e}", file=sys.stderr)
        elif file_name.endswith(".json") or file_name.endswith(".yaml"):
            src_file = os.path.join(input_dir, file_name)
            dest_file = os.path.join(output_dir, file_name)
            os.makedirs(os.path.dirname(dest_file), exist_ok=True)
            shutil.copy(src_file, dest_file)


def setup_pflotran_inputs(src_path: str, dest_path: str) -> None:
    """Copies required PFlotran input files (.ex and .uge) to the output directory."""
    target_dir = os.path.join(dest_path, "pflotran_inputs")
    os.makedirs(target_dir, exist_ok=True)

    files_to_copy = [f"{d}.ex" for d in ["east", "north", "south", "west"]] + ["mesh.uge"]

    for filename in files_to_copy:
        src_file = os.path.join(src_path, filename)
        if os.path.exists(src_file):
            shutil.copy(src_file, target_dir)
        else:
            log.info(f"Warning: Source file {filename} not found in {src_path}", file=sys.stderr)


def update_h5_keys(file_path: str, rename_map: dict[str, str]) -> None:
    """Renames datasets/groups in an HDF5 file recursively handling nested keys."""
    if not os.path.exists(file_path):
        return

    with h5py.File(file_path, "r+") as f:
        all_paths: list[str] = []
        f.visit(all_paths.append)
        # Sort by length descending to rename children before parents
        all_paths.sort(key=lambda x: -len(x))

        for old_path in all_paths:
            name = os.path.basename(old_path)
            if name in rename_map:
                new_name = rename_map[name]
                parent = os.path.dirname(old_path)
                new_path = os.path.join(parent, new_name) if parent else new_name
                # Use standard separators for HDF5 paths regardless of OS
                new_path = new_path.replace(os.sep, "/")

                try:
                    f.move(old_path, new_path)
                except (ValueError, RuntimeError):
                    # Fallback for complex object moves
                    f.copy(old_path, new_path)
                    del f[old_path]


def process_run_data(src_path: str, dest_path: str) -> None:
    """Copies and transforms run-specific output files and HDF5 data."""
    if not os.path.exists(src_path):
        return

    h5_renames = {
        "Permeability [m^2]": "Permeability X [m^2]",
        "Liquid X-Velocity [m_per_year]": "Liquid X-Velocity [m_per_y]",
        "Liquid Y-Velocity [m_per_year]": "Liquid Y-Velocity [m_per_y]",
        "Liquid Z-Velocity [m_per_year]": "Liquid Z-Velocity [m_per_y]",
    }

    for item in os.listdir(src_path):
        if item.startswith("datapoint-") and os.path.isdir(os.path.join(src_path, item)):
            run_id = item.split("-")[-1]
            run_dest = os.path.join(dest_path, f"RUN_{run_id}")
            os.makedirs(run_dest, exist_ok=True)

            # Copy and process pflotran.out and pflotran.h5
            for fname in ["pflotran.out", "pflotran.h5"]:
                src_file = os.path.join(src_path, item, fname)
                if os.path.exists(src_file):
                    dst_file = os.path.join(run_dest, fname)
                    shutil.copy(src_file, dst_file)
                    if fname.endswith(".h5"):
                        update_h5_keys(dst_file, h5_renames)

            # Copy permeability field
            perm_src = os.path.join(src_path, item, "permeability_field.h5")
            if os.path.exists(perm_src):
                shutil.copy(perm_src, os.path.join(run_dest, "interim_permeability_field.h5"))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        log.info("Usage: python script.py <input_path> <output_path>")
        sys.exit(1)

    input_dir, output_dir = sys.argv[1], sys.argv[2]

    log.info(f"Processing data from {input_dir} to {output_dir}...")

    convert_state_to_yaml(input_dir, output_dir)

    setup_pflotran_inputs(input_dir, output_dir)
    process_run_data(input_dir, output_dir)

    log.info("Workflow completed successfully.")
