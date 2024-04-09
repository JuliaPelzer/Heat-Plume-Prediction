import os
import pathlib
from dataclasses import dataclass
from typing import Dict

import yaml


def load_yaml(path: pathlib.Path, file_name="settings", **kwargs) -> Dict:
    with open(path / f"{file_name}.yaml", "r") as file:
        try:
            settings = yaml.safe_load(file, **kwargs)
        except:
            settings = yaml.load(file, **kwargs)
    return settings


def save_yaml(settings: Dict, path: str, name_file: str = "settings"):
    path = pathlib.Path(path)
    with open(path / f"{name_file}.yaml", "w") as file:
        yaml.dump(settings, file)

@dataclass
class SettingsTraining:
    inputs: str
    device: str
    epochs: int
    destination: pathlib.Path = ""
    dataset_raw: str = ""
    dataset_prep: str = ""
    case: str = "train"
    model: str = None
    visualize: bool = False
    save_inference: bool = False
    problem: str = "2stages"
    notes: str = ""
    dataset_train: str = ""
    dataset_val: str = ""
    dataset_test: str = "" 
    case_2hp: bool = False
    skip_per_dir: int = 4
    len_box: int = 256
    
    def __post_init__(self):
        if self.case_2hp:
            assert self.problem == "2stages", "2nd stage is only possible with 2stages problem"
        if self.case in ["finetune", "test"]:
            assert self.model is not None, "Path to model is not defined"
            assert self.model != "runs/default", "Please specify model path for testing or finetuning"

        if self.problem == "allin1":
            self.dataset_raw == ""
            self.dataset_prep = "" # TODO rm??
            self.case_2hp = False
        elif self.problem in ["2stages", "extend1", "extend2"]:
            self.dataset_train == ""
            self.dataset_val == ""
            self.dataset_test == ""

        if self.destination == "":
            if self.problem == "allin1":
                self.destination = f"{self.dataset_train} inputs_{self.inputs} case_{self.case} box{self.len_box} skip{self.skip_per_dir}"
            else:
                self.destination = f"{self.dataset_raw} inputs_{self.inputs} case_{self.case} box{self.len_box} skip{self.skip_per_dir}"

    def save(self):
        save_yaml(self.__dict__, self.destination, "command_line_arguments")
        
    def make_destination_path(self, destination_dir: pathlib.Path):
        if self.destination == "":
            self.destination = self.dataset_raw + " inputs_" + self.inputs + " case_"+self.case + " box"+str(self.len_box) + " skip"+str(self.skip_per_dir)
        self.destination = destination_dir / self.destination
        self.destination.mkdir(exist_ok=True)

    def make_model_path(self, destination_dir: pathlib.Path):
        self.model = destination_dir / self.model

    def save_notes(self):
        # save notes to text file in destination
        if self.notes != "":
            with open(self.destination / "notes.txt", "w") as file:
                file.write(self.notes)
