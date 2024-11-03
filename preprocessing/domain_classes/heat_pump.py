import logging
import pathlib
import sys
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from torch import long as torch_long
from torch import maximum, ones, save, tensor, unsqueeze, zeros_like, zeros, is_tensor, load, cat

from postprocessing.visualization import _aligned_colorbar
from data_stuff.transforms import SignedDistanceTransform
from networks.unet import UNet, UNetBC


class HeatPumpBox:
    def __init__(self, id, pos, orientation, inputs, names,corner_ll,corner_ur, dist_corner_hp=None, label=None, device="cpu"):
        self.id: str = id  # RUN_{ID}
        self.pos: list = tensor([int(pos[0]), int(pos[1])])  # (x,y), cell-ids
        self.orientation: float = float(orientation)
        self.dist_corner_hp: tensor = dist_corner_hp.to(dtype=torch_long)  # distance from corner of heat pump to corner of box
        self.inputs: tensor = inputs.to(device)  # extracted from large domain
        self.inputs_names: List[str] = names
        self.primary_temp_field: tensor = (ones(self.inputs[0].shape) * 10.6).to(device) #temperature field, calculated by 1HP-NN
        self.other_temp_field: tensor = (ones(self.inputs[0].shape) * 10.6).to(device)  # input for 2HP-NN
        self.output: tensor = (None)  # temperature field
        self.label = label.to(device)
        self.corner_ll: tensor = corner_ll
        self.corner_ur: tensor = corner_ur
        assert (self.pos[0] >= 0 and self.pos[1] >= 0), f"Heat pump position at {self.pos} is outside of domain"

    def recalc_sdf(self, info):
        # recalculate sdf per box (cant be done in prepare_dataset because of several hps in one domain)
        # TODO sizedependent... - works as long as boxes have same size in training as in prediction
        index_id = info["Inputs"]["Material ID"]["index"]
        index_sdf = info["Inputs"]["SDF"]["index"]
        assert self.inputs[index_id, self.dist_corner_hp[0], self.dist_corner_hp[1]] == 1, f"No HP at {self.pos}"
        self.inputs[index_sdf] = SignedDistanceTransform().sdf(self.inputs[index_id].detach().clone(), self.dist_corner_hp)
        assert (self.inputs[index_sdf].max() == 1 and self.inputs[index_sdf].min() == 0), "SDF not in [0,1]"

    def apply_nn(self, model: UNet, inputs:str="inputs", device:str="cpu"):
        if inputs == "inputs":
            input = unsqueeze(self.inputs, 0)
        elif inputs == "interim_outputs":
            input_tmp1 = self.primary_temp_field.unsqueeze(0)
            input_tmp2 = self.other_temp_field.unsqueeze(0)
            input = cat([input_tmp1, input_tmp2], dim=0).unsqueeze(0)
        input = input.to(device)
        output = model(input).squeeze().detach()
        return output
    
    def insert_extended_plume(self, output:tensor, insert_at:int, actual_len:int, device:str = "cpu"):
        if self.primary_temp_field.shape[0] < insert_at + actual_len:
            self.primary_temp_field = cat([self.primary_temp_field, zeros(insert_at + actual_len - self.primary_temp_field.shape[0], *self.primary_temp_field.shape[1:], device=device)])
        self.primary_temp_field[insert_at : insert_at+actual_len] = output[0, 0]

    def get_other_temp_field(self, single_hps):
        hp: HeatPumpBox
        for hp in single_hps:
            # get other hps
            if hp.id != self.id:
                assert (self.other_temp_field.shape == hp.primary_temp_field.shape), f"Shapes don't fit: {self.other_temp_field.shape} vs {hp.primary_temp_field.shape}"
                # get overlapping piece of 2nd hp T-box
                rel_pos = hp.pos - self.pos
                zeros2 = tensor([0, 0])
                offset = maximum(-rel_pos, zeros2).to(dtype=torch_long)
                end = tensor(hp.primary_temp_field.shape) - maximum(rel_pos, zeros2)
                tmp_2nd_hp = hp.primary_temp_field[offset[0] : end[0], offset[1] : end[1]]
                # insert at overlapping position in current hp
                offset2 = maximum(rel_pos, zeros2)
                end2 = tensor(self.other_temp_field.shape) - maximum(-rel_pos, zeros2)
                self.other_temp_field[offset2[0] : end2[0], offset2[1] : end2[1]] = maximum(self.other_temp_field[offset2[0] : end2[0], offset2[1] : end2[1]], tmp_2nd_hp)

    def save(self, run_id: str = "", dir: str = "HP-Boxes", additional_inputs: tensor = None, inputs_all: tensor = None, alt_label: tensor = None,):
        pathlib.Path(dir, "Inputs").mkdir(parents=True, exist_ok=True)
        pathlib.Path(dir, "Labels").mkdir(parents=True, exist_ok=True)
        # TODO NEXT
        if inputs_all != None:
            inputs = inputs_all
        elif is_tensor(additional_inputs):
            inputs = cat(self.inputs, additional_inputs, dim=0)
        else:
            inputs = self.inputs
        save(inputs, f"{dir}/Inputs/{run_id}HP_{self.id}.pt")
        if alt_label is None:
            save(self.label, f"{dir}/Labels/{run_id}HP_{self.id}.pt")
        else:
            save(unsqueeze(alt_label,0), f"{dir}/Labels/{run_id}HP_{self.id}.pt")

    def plot_and_reverse_norm(self, domain: "Domain", dir: pathlib.Path, data_to_plot: List[str] = None, names: List[str] = None, format_fig: str = "png"):
        if data_to_plot == None:
            data_to_plot = zeros_like(self.inputs)
            names = self.inputs_names.copy()
            for data_idx, data in enumerate(self.inputs):
                data_to_plot[data_idx] = domain.reverse_norm(data, property=names[data_idx])

            for data_idx, data in enumerate([self.primary_temp_field.unsqueeze(0), self.other_temp_field.unsqueeze(0), self.output.unsqueeze(0), self.label]):
                data_tmp = domain.reverse_norm(data, property="Temperature [C]")
                data_to_plot = cat((data_to_plot, data_tmp), dim=0)
            names += ["1HP-NN Prediction [C]", "Other Temperature Field [C]", "2HP-NN Prediction [C]", "Label [C]"]

        n_subplots = len(data_to_plot)
        assert len(data_to_plot) == len(names), "Number of data to plot does not match number of labels"
        plt.subplots(n_subplots, 1, sharex=True, figsize=(20, 3 * (n_subplots)))
        for idx in range(len(data_to_plot)):
            plt.subplot(n_subplots, 1, idx + 1)
            plt.imshow(data_to_plot[idx].T.cpu().detach().numpy())
            plt.gca().invert_yaxis()
            plt.xlabel("y [cells]")
            plt.ylabel("x [cells]")
            _aligned_colorbar(label=names[idx])
        plt.savefig(f"{dir}/hp_{self.id}.{format_fig}", format=format_fig)
        logging.warning(f"Saving plot to {dir}/hp_{self.id}.png")

    def measure_accuracy(self, domain:"Domain", plot_args: List = [False, "default.png"]):
        pred = self.output.cpu().detach().numpy()
        label = self.label.cpu().detach().numpy()
        pic_mae = abs(pred - label)
        pic_mse = (pred - label) ** 2

        if plot_args[0]:
            plt.figure()
            n_subplots = 6
            plt.subplots(n_subplots, 1, sharex=True, figsize=(20, 3 * (n_subplots)))

            plt.subplot(n_subplots, 1, 1)
            plt.imshow(self.primary_temp_field.cpu().detach().numpy().T)
            plt.gca().invert_yaxis()
            plt.xlabel("x [cells]")
            plt.ylabel("y [cells]")
            _aligned_colorbar(label="In: 1HP-Prediction")

            plt.subplot(n_subplots, 1, 2)
            plt.imshow(self.other_temp_field.cpu().detach().numpy().T)
            plt.gca().invert_yaxis()
            plt.xlabel("x [cells]")
            plt.ylabel("y [cells]")
            _aligned_colorbar(label="In: Overlap")

            plt.subplot(n_subplots, 1, 3)
            plt.imshow(pred.T)
            plt.gca().invert_yaxis()
            plt.xlabel("x [cells]")
            plt.ylabel("y [cells]")
            _aligned_colorbar(label="Prediction: Combination")

            plt.subplot(n_subplots, 1, 4)
            plt.imshow(label.T)
            plt.gca().invert_yaxis()
            plt.xlabel("x [cells]")
            plt.ylabel("y [cells]")
            _aligned_colorbar(label="Label: True Combination")

            plt.subplot(n_subplots, 1, 5)
            plt.imshow(pic_mae.T)
            plt.gca().invert_yaxis()
            plt.xlabel("x [cells]")
            plt.ylabel("y [cells]")
            _aligned_colorbar(label="Error: MAE")

            plt.subplot(n_subplots, 1, 6)
            plt.imshow(pic_mse.T)
            plt.gca().invert_yaxis()
            plt.xlabel("x [cells]")
            plt.ylabel("y [cells]")
            _aligned_colorbar(label="Error: RMSE")
            
            plt.savefig(plot_args[1])
        return np.mean(pic_mae), np.mean(pic_mse)