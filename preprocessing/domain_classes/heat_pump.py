from typing import List, Union
from torch import long as torch_long
from torch import ones, tensor, unsqueeze, zeros, cat

from preprocessing.transforms import SignedDistanceTransform
from processing.networks.model import Model
from processing.networks.CdMLP import CdMLP


class HeatPumpBox:
    def __init__(self, id, pos, orientation, inputs, names, dist_corner_hp=None, label=None, device="cpu"):
        self.id: str = id  # RUN_{ID}
        self.pos: list = tensor([int(pos[0]), int(pos[1])])  # (x,y), cell-ids
        self.orientation: float = float(orientation)
        self.dist_corner_hp: tensor = dist_corner_hp.to(dtype=torch_long)  # distance from corner of heat pump to corner of box
        self.inputs: tensor = inputs.to(device)  # extracted from large domain
        self.inputs_names: List[str] = names
        self.primary_temp_field: tensor = (None)  #temperature field, calculated by 1HP-NN
        self.other_temp_field: tensor = (ones(self.inputs[0].shape) * 10.6).to(device)  # input for 2HP-NN
        self.output: tensor = (None)  # temperature field
        self.label = label.to(device)
        assert (self.pos[0] >= 0 and self.pos[1] >= 0), f"Heat pump position at {self.pos} is outside of domain"

    def recalc_sdf(self, info):
        # recalculate sdf per box (cant be done in prepare_dataset because of several hps in one domain)
        # TODO sizedependent... - works as long as boxes have same size in training as in prediction
        index_id = info["Inputs"]["Material ID"]["index"]
        index_sdf = info["Inputs"]["SDF"]["index"]
        assert self.inputs[index_id, self.dist_corner_hp[0], self.dist_corner_hp[1]] == 1, f"No HP at {self.pos}"
        self.inputs[index_sdf] = SignedDistanceTransform().sdf(self.inputs[index_id].detach().clone(), self.dist_corner_hp)
        assert (self.inputs[index_sdf].max() == 1 and self.inputs[index_sdf].min() == 0), "SDF not in [0,1]"

    def apply_nn(self, case_model:str, model: Union[Model, CdMLP], inputs:str="inputs", device:str="cpu", info:dict=None):
        if case_model == "unet":
            if inputs == "inputs":
                input = unsqueeze(self.inputs, 0)
            elif inputs == "interim_outputs":
                input_tmp1 = self.primary_temp_field.unsqueeze(0)
                input_tmp2 = self.other_temp_field.unsqueeze(0)
                input = cat([input_tmp1, input_tmp2], dim=0).unsqueeze(0)
            output = model.infer(input, device).squeeze().detach()
        elif case_model == "cdmlp":
            train_height, train_width = info["CellsNumber"]
            print(train_height, train_width)
            output = model.apply(self.inputs.unsqueeze(0), self.pos.unsqueeze(0), training_height=train_height, training_width=train_width).squeeze()
            output = tensor(output).to(device)
            self.dist_corner_hp -= 2

        return output
    
    def get_global_corner_ll(self):
        return self.pos - self.dist_corner_hp
    
    def insert_extended_plume(self, output:tensor, insert_at:int, actual_len:int, device:str = "cpu"):
        if self.primary_temp_field.shape[0] < insert_at + actual_len:
            self.primary_temp_field = cat([self.primary_temp_field, zeros(insert_at + actual_len - self.primary_temp_field.shape[0], *self.primary_temp_field.shape[1:], device=device)])
        self.primary_temp_field[insert_at : insert_at+actual_len] = output[0, 0]