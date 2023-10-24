import logging
import pathlib
import sys
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from torch import tensor, save, load, unsqueeze, ones, cat, maximum
from torch import long as torch_long

from data_stuff.transforms import SignedDistanceTransform
from utils.visualization import _aligned_colorbar


class HeatPump:
    def __init__(self, id, pos, orientation, inputs, dist_corner_hp=None, label=None, device="cpu"):
        self.id: str = id  # RUN_{ID}
        self.pos: list = tensor([int(pos[0]), int(pos[1])])  # (x,y), cell-ids
        self.orientation: float = float(orientation)
        self.dist_corner_hp: tensor = dist_corner_hp.to(dtype=torch_long)  # distance from corner of heat pump to corner of box
        self.inputs: tensor = inputs.to(device)  # extracted from large domain
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

    def apply_nn(self, model, inputs:str="inputs"):
        if inputs == "inputs":
            input = unsqueeze(self.inputs, 0)
        elif inputs == "interim_outputs":
            input = unsqueeze(tensor(np.array([self.primary_temp_field, self.other_temp_field])), 0) # TODO tensor??
        model.eval()
        output = model(input)
        output = output.squeeze().detach()
        return output

    def get_other_temp_field(self, single_hps):
        hp: HeatPump
        for hp in single_hps:
            # get other hps
            if hp.id != self.id:
                assert (self.other_temp_field.shape == hp.primary_temp_field.shape), "Shapes don't fit"
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

    def save(self, run_id: str = "", dir: str = "HP-Boxes", additional_inputs: tensor = None, inputs_all: tensor = None,):
        # dir_in = dir / "Inputs"
        # dir_in.mkdir(parents=True, exist_ok=True)
        pathlib.Path(dir, "Inputs").mkdir(parents=True, exist_ok=True)
        pathlib.Path(dir, "Labels").mkdir(parents=True, exist_ok=True)
        # TODO NEXT
        if inputs_all != None:
            inputs = inputs_all
        elif (additional_inputs != None).any():
            inputs = cat(self.inputs, additional_inputs, dim=0)
        else:
            inputs = self.inputs
        save(inputs, f"{dir}/Inputs/{run_id}HP_{self.id}.pt")
        save(self.label, f"{dir}/Labels/{run_id}HP_{self.id}.pt")

    def save_pred(self, run_id: str = "", dir: str = "HP-Boxes", additional_inputs: np.ndarray = None, inputs_all: np.ndarray = None,):
        dir_in = dir / "Inputs"
        dir_in.mkdir(parents=True, exist_ok=True)
        if (inputs_all != None): #.any():
            inputs = inputs_all
        elif (additional_inputs != None).any():
            inputs = np.append(self.inputs, additional_inputs, axis=0)
        else:
            inputs = self.inputs
        save(inputs, f"{dir}/{run_id}HP_{self.id}_prediction.pt")
        save(self.label, f"{dir}/{run_id}HP_{self.id}_label.pt")

    def plot_fields(self, n_subplots: int, domain: "Domain"):
        plt.subplots(n_subplots, 1, sharex=True, figsize=(20, 3 * (n_subplots)))
        idx = 1
        for input_idx, input in enumerate(self.inputs):
            plt.subplot(n_subplots, 1, idx)
            plt.imshow(input.T)
            plt.gca().invert_yaxis()
            plt.xlabel("y [cells]")
            plt.ylabel("x [cells]")
            _aligned_colorbar(label=domain.get_name_from_index(input_idx))
            idx += 1

    def plot_prediction_1HP(self, n_subplots, idx):
        plt.subplot(n_subplots, 1, idx)
        plt.imshow(self.primary_temp_field.T)
        plt.gca().invert_yaxis()
        plt.xlabel("y [cells]")
        plt.ylabel("x [cells]")
        _aligned_colorbar(label="Temperature [C]")

    def plot_prediction_2HP(self, n_subplots, idx):
        plt.subplot(n_subplots, 1, idx)
        plt.imshow(self.output.T)
        plt.gca().invert_yaxis()
        plt.xlabel("y [cells]")
        plt.ylabel("x [cells]")
        _aligned_colorbar(label="Temperature [C]")

    def plot_1HP(self, domain: "Domain", dir: str = "HP-Boxes"):
        n_subplots = len(self.inputs) + 1
        self.plot_fields(n_subplots, domain)
        self.plot_prediction_1HP(n_subplots, idx=n_subplots)
        logging.info(f"Saving plot to {dir}/hp_{self.id}.png")
        plt.savefig(f"{dir}/hp_{self.id}.png")

    def plot_2HP(self, domain: "Domain", dir: str = "HP-Boxes_2HP"):
        n_subplots = len(self.inputs) + 2
        self.plot_fields(n_subplots, domain)
        self.plot_prediction_1HP(n_subplots, idx=n_subplots - 1)
        self.plot_prediction_2HP(n_subplots, idx=n_subplots)
        plt.savefig(f"{dir}/hp_{self.id}.png")

    def plot(
        self,
        dir: str = "HP-Boxes",
        data_to_plot: np.ndarray = None,
        names: np.ndarray = None,
    ):
        if data_to_plot.any() != None:
            n_subplots = len(data_to_plot)
            assert len(data_to_plot) == len(
                names
            ), "Number of data to plot does not match number of labels"
            plt.subplots(n_subplots, 1, sharex=True, figsize=(20, 3 * (n_subplots)))
            for idx in range(len(data_to_plot)):
                plt.subplot(n_subplots, 1, idx + 1)
                plt.imshow(data_to_plot[idx].T)
                plt.gca().invert_yaxis()
                plt.xlabel("y [cells]")
                plt.ylabel("x [cells]")
                _aligned_colorbar(label=names[idx])
            plt.savefig(f"{dir}/hp_{self.id}.png")
            logging.warning(f"Saving plot to {dir}/hp_{self.id}.png")
        else:
            logging.warning("No data to plot given")

    def measure_accuracy(self, domain:"Domain", plot_args: List = [False, "default.png"]):
        pred = self.output
        label = domain.reverse_norm(self.label, property="Temperature [C]")[0]
        pic_mae = abs(pred - label)
        pic_mse = abs(pred - label) ** 2

        if plot_args[0]:
            plt.figure()
            n_subplots = 6
            plt.subplots(n_subplots, 1, sharex=True, figsize=(20, 3 * (n_subplots)))

            plt.subplot(n_subplots, 1, 1)
            plt.imshow(self.primary_temp_field.T)
            plt.gca().invert_yaxis()
            plt.xlabel("x [cells]")
            plt.ylabel("y [cells]")
            _aligned_colorbar(label="In: 1HP-Prediction [C]")

            plt.subplot(n_subplots, 1, 2)
            plt.imshow(self.other_temp_field.T)
            plt.gca().invert_yaxis()
            plt.xlabel("x [cells]")
            plt.ylabel("y [cells]")
            _aligned_colorbar(label="In: Overlap [C]")

            plt.subplot(n_subplots, 1, 3)
            plt.imshow(pred.T)
            plt.gca().invert_yaxis()
            plt.xlabel("x [cells]")
            plt.ylabel("y [cells]")
            _aligned_colorbar(label="Prediction: Combination [C]")

            plt.subplot(n_subplots, 1, 4)
            plt.imshow(label.T)
            plt.gca().invert_yaxis()
            plt.xlabel("x [cells]")
            plt.ylabel("y [cells]")
            _aligned_colorbar(label="Label: True Combination [C]")

            plt.subplot(n_subplots, 1, 5)
            plt.imshow(pic_mae.T)
            plt.gca().invert_yaxis()
            plt.xlabel("x [cells]")
            plt.ylabel("y [cells]")
            _aligned_colorbar(label="Error: MAE [C]")

            plt.subplot(n_subplots, 1, 6)
            plt.imshow(pic_mse.T)
            plt.gca().invert_yaxis()
            plt.xlabel("x [cells]")
            plt.ylabel("y [cells]")
            _aligned_colorbar(label="Error: MSE [C]")
            
            plt.savefig(plot_args[1])
        return np.mean(pic_mae), np.mean(pic_mse)