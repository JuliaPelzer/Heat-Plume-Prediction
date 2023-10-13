from dataclasses import dataclass
import os
import pathlib
import typing
import yaml
from utils.utils import re_split_number_text

# Data classes for paths
@dataclass
class Paths1HP:
    raw_dir: pathlib.Path # boxes
    datasets_prepared_dir: pathlib.Path
    dataset_1st_prep_path: pathlib.Path
    destination_dir: pathlib.Path

@dataclass
class Paths2HP:
    raw_dir: pathlib.Path # domain
    dataset_model_trained_with_prep_path: pathlib.Path # 1hp-boxes
    dataset_1st_prep_path: pathlib.Path # domain
    model_1hp_path: pathlib.Path
    datasets_prepared_dir: pathlib.Path # 2hp-boxes
    datasets_boxes_prep_path: pathlib.Path # 2hp-boxes
    destination_dir: pathlib.Path
    model_2hp_path: typing.Optional[pathlib.Path] = None

# Functions for setting paths
def set_paths_1hpnn(dataset_name: str, inputs:str = "")-> Paths1HP:
    paths_file = "paths.yaml"
    if not os.path.exists(paths_file):
        raise FileNotFoundError(f"{paths_file} not found")

    with open(paths_file, "r") as f:
        paths = yaml.safe_load(f)

    default_raw_dir = pathlib.Path(paths["default_raw_dir"])
    destination_dir = pathlib.Path(paths["models_1hp_dir"])
    datasets_prepared_dir = pathlib.Path(paths["datasets_prepared_dir"])
    dataset_prep = f"{dataset_name} inputs_{inputs}"
    dataset_prepared_full_path = datasets_prepared_dir / dataset_prep

    return Paths1HP(default_raw_dir, datasets_prepared_dir, dataset_prepared_full_path, destination_dir), dataset_prep

def set_paths_2hpnn(dataset_name: str, preparation_case: str, model_name_2hp: str = None):
    paths_file = "paths.yaml"
    
    if not os.path.exists(paths_file):
        raise FileNotFoundError(f"{paths_file} not found")
    with open(paths_file, "r") as f:
        paths = yaml.safe_load(f)

    datasets_raw_domain_dir = pathlib.Path(paths["datasets_raw_domain_dir"])
    datasets_prepared_domain_dir = pathlib.Path(paths["datasets_prepared_domain_dir"])
    prepared_1hp_dir = pathlib.Path(paths["prepared_1hp_best_models_and_data_dir"])
    destination_dir = pathlib.Path(paths["models_2hp_dir"])
    datasets_prepared_2hp_dir = pathlib.Path(paths["datasets_prepared_dir_2hp"])
    check_validity_preparation(preparation_case)

    prepared_1hp_dir = prepared_1hp_dir / preparation_case
    for path in prepared_1hp_dir.iterdir():
        if path.is_dir():
            if "current" in path.name: # TODO change to "model"
                model_1hp_path = prepared_1hp_dir / path.name
            elif "dataset" in path.name:
                dataset_model_trained_with_prep_path = prepared_1hp_dir / path.name
    
    inputs = re_split_number_text(str(preparation_case))[0]
    dataset_1st_prep_path = datasets_prepared_domain_dir / f"{dataset_name} inputs_{inputs}"
    dataset_prep_2hp_path = f"{dataset_name} inputs_{preparation_case} boxes"
    datasets_boxes_prep_path = datasets_prepared_2hp_dir / dataset_prep_2hp_path
    model_2hp_path = destination_dir / model_name_2hp if model_name_2hp is not None else None

    return Paths2HP(
        datasets_raw_domain_dir,
        dataset_model_trained_with_prep_path,
        dataset_1st_prep_path,
        model_1hp_path,
        datasets_prepared_2hp_dir,
        datasets_boxes_prep_path,
        destination_dir,
        model_2hp_path
        ), inputs, dataset_prep_2hp_path

def check_validity_preparation(preparation_case:str):
    # Check that preparation_case is valid
    assert preparation_case in ["gksi100", "ogksi1000", "gksi1000", "pksi100", "pksi1000", "ogksi1000_finetune", "gki100"], "preparation_case must be one of ['gksi100', 'ogksi1000', 'gksi1000', 'pksi100', 'pksi1000', 'ogksi1000_finetune', 'gki100']"
    