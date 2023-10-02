import os
import pathlib
from dataclasses import dataclass
from typing import Dict
import yaml


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
    name_folder_destination: str
    datasets_path: str = "/home/pelzerja/Development/dataset_generation_pflotran/Phd_simulation_groundtruth/datasets"
    case: str = "train"
    finetune: bool = False
    path_to_model: str = None
    test: bool = False
    case_2hp: bool = False
    
    def __post_init__(self):
        if not self.case_2hp: 
            self.dataset_name += " inputs_"+self.inputs_prep

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

        self.set_destination()
        self.make_destination_dir()
        self.save()

    def save(self):
        save_yaml(self.__dict__, os.path.join(
            "runs", self.name_folder_destination), "settings_training")
        
    def set_destination(self):
        if self.name_folder_destination == "":
            self.name_folder_destination = self.dataset_name + " case_" + self.case

    def make_destination_dir(self):
        destination_dir = pathlib.Path(os.getcwd(), "runs", self.name_folder_destination)
        destination_dir.mkdir(parents=True, exist_ok=True)

@dataclass
class SettingsPrepare:
    raw_dir: str
    datasets_dir: str
    dataset_name: str
    inputs_prep: str