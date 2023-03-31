import os
from typing import Dict
import yaml
from torch import Tensor
from dataclasses import dataclass
import pathlib

def load_settings(path: str, file_name="settings") -> Dict:
    path = pathlib.Path(path)
    with open(path.joinpath(f"{file_name}.yaml"), "r") as file:
        settings = yaml.load(file)
    return settings


def save_settings(settings: Dict, path: str, name_file: str = "settings"):
    path = pathlib.Path(path)
    with open(path.joinpath(f"{name_file}.yaml"), "w") as file:
        yaml.dump(settings, file)

@dataclass
class SettingsTraining:
    dataset_name: str
    device: str
    epochs: int
    model_choice: str
    finetune: bool
    path_to_model: str
    name_folder_destination: str
    datasets_path: str = "/home/pelzerja/Development/simulation_groundtruth_pflotran/Phd_simulation_groundtruth/datasets"

    def __post_init__(self):
        self.path_to_model = os.path.join("runs", self.path_to_model)

    def save(self):
        save_settings(self.__dict__, os.path.join(
            "runs", self.name_folder_destination), "settings_training")