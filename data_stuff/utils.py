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
    dataset_raw: str
    inputs: str
    device: str
    epochs: int
    destination_dir: str 
    datasets_dir: str = ""
    dataset_prep: str = ""
    case: str = "test"
    model: str = None
    test: bool = False
    visualize: bool = False
    
    def __post_init__(self):

        self.model = os.path.join("runs", self.model)
        self.set_destination()
        self.make_destination_dir()

    def save(self):
        save_yaml(self.__dict__, os.path.join(
            "runs", self.destination_dir), "command_line_arguments")
        
    def set_destination(self):
        if self.destination_dir == "":
            if not self.case_2hp:
                extension = " inputs_"+self.inputs + " case_"+self.case
            else:
                extension = " case_"+self.case
                # TODO
            self.destination_dir = self.dataset_raw + extension

    def make_destination_dir(self):
        destination_dir = pathlib.Path(os.getcwd(), "runs", self.destination_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)