import os
import pathlib
from dataclasses import dataclass
from typing import Dict

import yaml
from torch import Tensor


def load_yaml(path: str, file_name="settings") -> Dict:
    path = pathlib.Path(path)
    with open(path.joinpath(f"{file_name}.yaml"), "r") as file:
        settings = yaml.safe_load(file)
    return settings


def save_yaml(settings: Dict, path: str, name_file: str = "settings"):
    path = pathlib.Path(path)
    with open(path.joinpath(f"{name_file}.yaml"), "w") as file:
        yaml.dump(settings, file)

@dataclass
class SettingsTraining:
    dataset_name: str
    inputs_prep: str
    device: str
    epochs: int
    model_choice: str
    name_folder_destination: str
    datasets_path: str = "/home/pelzerja/Development/dataset_generation_pflotran/Phd_simulation_groundtruth/datasets"
    case: str = "train"
    finetune: bool = False
    path_to_model: str = None
    test: bool = False
    name_extension: str = "" # extension of dataset name, e.g. _grad_p
    case_2hp: bool = False
    loss: str = "data"
    
    def __post_init__(self):
        self.path_to_model = os.path.join("runs", self.path_to_model)
        if self.case in ["finetune", "finetuning", "Finetune", "Finetuning"]:
            self.finetune = True
            self.case = "finetune"
            assert self.path_to_model is not None, "Path to model is not defined"
        elif self.case in ["test", "testing", "Test", "Testing", "TEST"]:
            self.case = "test"
            self.test = True
            assert self.finetune is False, "Finetune is not possible in test mode"
        elif self.case in ["train", "training", "Train", "Training", "TRAIN"]:
            self.case = "train"
            assert self.finetune is False, "Finetune is not possible in train mode"
            assert self.test is False, "Test is not possible in train mode"

    def save(self):
        save_yaml(self.__dict__, os.path.join(
            "runs", self.name_folder_destination), "settings_training")
        
@dataclass
class SettingsPrepare:
    raw_dir: str
    datasets_dir: str
    dataset_name: str
    inputs_prep: str
    name_extension: str = ""