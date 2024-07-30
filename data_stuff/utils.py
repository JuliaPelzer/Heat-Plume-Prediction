import os
import pathlib
from dataclasses import dataclass
from typing import Dict
import yaml


def load_yaml(path: pathlib.Path, file_name="settings") -> Dict:
    with open(path / f"{file_name}.yaml", "r") as file:
        settings = yaml.safe_load(file)
    return settings


def save_yaml(settings: Dict, path: str, name_file: str = "settings"):
    path = pathlib.Path(path)
    with open(path / f"{name_file}.yaml", "w") as file:
        yaml.dump(settings, file)

@dataclass
class SettingsTraining:
    dataset_raw: str
    inputs: str
    device: str
    epochs: int
    last_cell_mode: str
    destination: pathlib.Path = ""
    dataset_prep: str = ""
    case: str = "train"
    finetune: bool = False
    model: str = None
    test: bool = False
    case_2hp: bool = False
    visualize: bool = False
    save_inference: bool = False
    problem: str = "2stages"
    notes: str = ""
    skip_per_dir: int = 4
    len_box: int = 640
    net: str = "convLSTM"
    total_time_steps: int = 20
    time_step_to_predict: int = 10
    vis_entire_plume: bool = False
    
    def __post_init__(self):
        if self.case in ["finetune", "finetuning", "Finetune", "Finetuning"]:
            self.finetune = True
            self.case = "finetune"
            assert self.model is not None, "Path to model is not defined"
        elif self.case in ["test", "testing", "Test", "Testing", "TEST"]:
            self.case = "test"
            self.test = True
            assert self.finetune is False, "Finetune is not possible in test mode"
        elif self.case in ["train", "training", "Train", "Training", "TRAIN"]:
            self.case = "train"
            assert self.finetune is False, "Finetune is not possible in train mode"
            assert self.test is False, "Test is not possible in train mode"

        if self.case in ["test", "finetune"]:
            assert self.model != "runs/default", "Please specify model path for testing or finetuning"

        if self.destination == "":
            self.destination = f"case_{self.case} net_{self.net} steps_{self.total_time_steps} box{self.len_box} skip{self.skip_per_dir} last_cell_mode_{self.last_cell_mode} 2"

    def save(self):
        save_yaml(self.__dict__, self.destination, "command_line_arguments")
        
    def make_destination_path(self, destination_dir: pathlib.Path):
        if self.destination == "":
            self.destination = f"case_{self.case} net_{self.net} steps_{self.total_time_steps} box{self.len_box} skip{self.skip_per_dir} last_cell_mode_{self.last_cel_mode} 2"
        self.destination = destination_dir / self.destination
        self.destination.mkdir(parents=True, exist_ok=True)

    def make_model_path(self, destination_dir: pathlib.Path):
        self.model = destination_dir / self.model

    def save_notes(self):
        # save notes to text file in destination
        if self.notes != "":
            with open(self.destination / "notes.txt", "w") as file:
                file.write(self.notes)