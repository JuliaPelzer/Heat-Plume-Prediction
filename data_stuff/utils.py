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
    def __init__(self,
                 dataset_raw: str,
                 inputs: str,
                 device: str,
                 epochs: int,
                 prev_boxes: int,
                 extend: int,
                 destination: pathlib.Path = "",
                 dataset_prep: str = "",
                 case: str = "train",
                 finetune: bool = False,
                 model: str = None,
                 test: bool = False,
                 case_2hp: bool = False,
                 visualize: bool = False,
                 save_inference: bool = False,
                 problem: str = "2stages",
                 notes: str = "",
                 skip_per_dir: int = 4,
                 len_box: int = 640,
                 net: str = "convLSTM",
                 vis_entire_plume: bool = False,
                 overfit: int = 0,
                 num_layers: int = 1,
                 loss: str = 'mse',
                 activation: str = 'relu',
                 enc_conv_features = [16, 32, 64, 64, 64],
                 dec_conv_features = [64, 64, 64],
                 enc_kernel_sizes = [7, 5, 5, 5, 5],
                 dec_kernel_sizes = [5, 5, 7]):

        self.dataset_raw = dataset_raw
        self.inputs = inputs
        self.device = device
        self.epochs = epochs
        self.prev_boxes = prev_boxes
        self.extend = extend
        self.destination = destination
        self.dataset_prep = dataset_prep
        self.case = case
        self.finetune = finetune
        self.model = model
        self.test = test
        self.case_2hp = case_2hp
        self.visualize = visualize
        self.save_inference = save_inference
        self.problem = problem
        self.notes = notes
        self.skip_per_dir = skip_per_dir
        self.len_box = len_box
        self.net = net
        self.vis_entire_plume = vis_entire_plume
        self.overfit = overfit
        self.num_layers = num_layers
        self.loss = loss
        self.activation = activation
        self.enc_conv_features = enc_conv_features
        self.dec_conv_features = dec_conv_features
        self.enc_kernel_sizes = enc_kernel_sizes
        self.dec_kernel_sizes = dec_kernel_sizes 
    
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

        if self.overfit == 0:
            self.overfit_str = ""
        else:
            self.overfit_str = f' overfit_{self.overfit}'

        if self.destination == "": 
            self.destination = f"case_{self.case} prev_{self.prev_boxes} extend_{self.extend} skip_{self.skip_per_dir} f_32 {self.loss} {self.num_layers}layers"

        

    def save(self):
        save_yaml(self.__dict__, self.destination, "command_line_arguments")
        
    def make_destination_path(self, destination_dir: pathlib.Path):
        if self.destination == "":
            self.destination = f"case_{self.case} prev_{self.prev_boxes} extend_{self.extend} skip_{self.skip_per_dir} f_32 {self.loss} {self.num_layers}layers"
        self.destination = destination_dir / self.destination
        self.destination.mkdir(parents=True, exist_ok=True)

    def make_model_path(self, destination_dir: pathlib.Path):
        self.model = destination_dir / self.model

    def save_notes(self):
        # save notes to text file in destination
        if self.notes is not None:
            with open(self.destination / "notes.txt", "w") as file:
                file.write(self.notes)