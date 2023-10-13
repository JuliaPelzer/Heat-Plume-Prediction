import logging
import os
import pathlib
import shutil
import sys
from math import cos, sin

import matplotlib.pyplot as plt
from torch import load, save, tensor, ones, where, max, squeeze, stack
from torch import long as torch_long

sys.path.append("/home/pelzerja/pelzerja/test_nn/1HP_NN")  # relevant for remote
sys.path.append("/home/pelzerja/Development/1HP_NN")  # relevant for local
from data_stuff.utils import load_yaml
from preprocessing.prepare_1ststage import expand_property_names
from utils.utils import beep
from utils.visualization import _aligned_colorbar

from domain_classes.heat_pump import HeatPump
from domain_classes.stitching import Stitching


class Domain:
    def __init__(
        self, info_path: str, stitching_method: str = "max", file_name: str = "RUN_0.pt", device = "cpu"):
        self.skip_datapoint = False
        self.info = load_yaml(info_path, "info")
        self.size: tuple[int, int] = [
            self.info["CellsNumber"][0],
            self.info["CellsNumber"][1],
        ]  # (x, y), cell-ids
        self.background_temperature: float = 10.6
        self.inputs: tensor = self.load_datapoint(info_path, case="Inputs", file_name=file_name)
        self.label: tensor = self.load_datapoint(info_path, case="Labels", file_name=file_name)
        self.prediction: tensor = (ones(self.size) * self.background_temperature).to(device)
        self.prediction_2HP: tensor = (ones(self.size) * self.background_temperature).to(device)
        self.stitching: Stitching = Stitching(stitching_method, self.background_temperature)
        self.normed: bool = True
        self.file_name: str = file_name
        if (self.get_input_field_from_name("Permeability X [m^2]").max() > 1
            or self.get_input_field_from_name("Permeability X [m^2]").min() < 0):
            print(f"Permeability X [m^2] not in range (0,1) for {file_name} but at ({self.get_input_field_from_name('Permeability X [m^2]').max()}, {self.get_input_field_from_name('Permeability X [m^2]').min()})")
            origin_2hp_prep = info_path
            pathlib.Path(origin_2hp_prep, "unusable/Inputs").mkdir(parents=True, exist_ok=True)
            pathlib.Path(origin_2hp_prep, "unusable/Labels").mkdir(parents=True, exist_ok=True)
            shutil.move(
                pathlib.Path(origin_2hp_prep, "Inputs", file_name),
                pathlib.Path(origin_2hp_prep, "unusable", "Inputs", f"{file_name.split('.')[0]}_k_outside_0_1.pt",),)
            shutil.move(
                pathlib.Path(origin_2hp_prep, "Labels", file_name),
                pathlib.Path(origin_2hp_prep, "unusable", "Labels", file_name),)
            self.skip_datapoint=True
        # assert (
        #     self.get_input_field_from_name("Permeability X [m^2]").max() <= 1
        # ), f"Max of permeability X [m^2] not < 1 but {self.get_input_field_from_name('Permeability X [m^2]').max()} for {file_name}"
        # assert (
        #     self.get_input_field_from_name("Permeability X [m^2]").min() >= 0
        # ), f"Min of permeability X [m^2] not > 0 but {self.get_input_field_from_name('Permeability X [m^2]').min()} for {file_name}"
        # TODO : wenn perm/pressure nicht mehr konstant sind, muss dies zu den HP-Boxen verschoben werden
        else:
            try:
                p_related_name = "Pressure Gradient [-]"
                p_related_field = self.get_input_field_from_name(p_related_name)
            except:
                p_related_name = "Liquid Pressure [Pa]"
                p_related_field = self.get_input_field_from_name(p_related_name)
            logging.info(
                f"{p_related_name} in range ({p_related_field.max()}, {p_related_field.min()})")

            if p_related_field.max() > 1 or p_related_field.min() < 0:
                print(f"{p_related_name} not in range (0,1) for {file_name} but at ({p_related_field.max()}, {p_related_field.min()})")
                origin_2hp_prep = info_path
                pathlib.Path(origin_2hp_prep, "unusable/Inputs").mkdir(parents=True, exist_ok=True)
                pathlib.Path(origin_2hp_prep, "unusable/Labels").mkdir(parents=True, exist_ok=True)

                shutil.move(
                    pathlib.Path(origin_2hp_prep, "Inputs", file_name),
                    pathlib.Path(origin_2hp_prep, "unusable", "Inputs", f"{file_name.split('.')[0]}_p_outside_0_1.pt",),)
                shutil.move(
                    pathlib.Path(origin_2hp_prep, "Labels", file_name),
                    pathlib.Path(origin_2hp_prep, "unusable", "Labels", file_name),)
                beep()
                self.skip_datapoint=True
            # assert (
            #     p_related_field.max() <= 1 and p_related_field.min() >= 0
            # ), f"{p_related_name} not in range (0,1) but {p_related_field.max(), p_related_field.min()}"

    def save(self, folder: str = "", name: str = "test"):
        pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
        save(self.prediction, os.path.join(folder, f"{name}prediction.pt"))
        save(self.label, os.path.join(folder, f"{name}label.pt"))
        save(self.inputs, os.path.join(folder, f"{name}inputs.pt"))

    def load_datapoint(
        self, dataset_domain_path: str, case: str = "Inputs", file_name="RUN_0.pt"
    ):
        # load dataset of large domain
        file_path = os.path.join(dataset_domain_path, case, file_name)
        data = load(file_path)
        return data

    def get_index_from_name(self, name: str):
        return self.info["Inputs"][name]["index"]

    def get_name_from_index(self, index: int):
        for property, values in self.info["Inputs"].items():
            if values["index"] == index:
                return property

    def get_input_field_from_name(self, name: str):
        field_idx = self.get_index_from_name(name)
        field = self.inputs[field_idx, :, :]
        return field

    def norm(self, data: tensor, property: str = "Temperature [C]"):
        norm_fct, max_val, min_val, mean_val, std_val = self.get_norm_info(property)

        if norm_fct == "Rescale":
            out_min, out_max = (
                0,
                1,
            )  # TODO Achtung! Hardcoded, values same as in transforms.NormalizeTransform.out_min/max
            delta = max_val - min_val
            data = (data - min_val) / delta * (out_max - out_min) + out_min
        elif norm_fct == "Standardize":
            data = (data - mean_val) / std_val
        elif norm_fct is None:
            pass
        else:
            raise ValueError(f"Normalization type '{self.norm['Norm']}' not recognized")
        return data

    def reverse_norm(self, data: tensor, property: str = "Temperature [C]"):
        norm_fct, max_val, min_val, mean_val, std_val = self.get_norm_info(property)

        if norm_fct == "Rescale":
            out_min, out_max = (
                0,
                1,
            )  # TODO Achtung! Hardcoded, values same as in transforms.NormalizeTransform.out_min/max
            delta = max_val - min_val
            data = (data - out_min) / (out_max - out_min) * delta + min_val
        elif norm_fct == "Standardize":
            data = data * std_val + mean_val
        elif norm_fct is None:
            pass
        else:
            raise ValueError(
                f"Normalization type '{self.norm_fct['Norm']}' not recognized"
            )
        return data

    def get_norm_info(self, property: str = "Temperature [C]"):
        try:
            norm_fct = self.info["Inputs"][property]["norm"]
            max_val = self.info["Inputs"][property]["max"]
            min_val = self.info["Inputs"][property]["min"]
            mean_val = self.info["Inputs"][property]["mean"]
            std_val = self.info["Inputs"][property]["std"]
        except:
            norm_fct = self.info["Labels"][property]["norm"]
            max_val = self.info["Labels"][property]["max"]
            min_val = self.info["Labels"][property]["min"]
            mean_val = self.info["Labels"][property]["mean"]
            std_val = self.info["Labels"][property]["std"]
        return norm_fct, max_val, min_val, mean_val, std_val

    def extract_hp_boxes(self, device:str = "cpu") -> list:
        # TODO decide: get hp_boxes based on grad_p or based on v or get squared boxes around hp
        material_ids = self.get_input_field_from_name("Material ID")
        size_hp_box = tensor([self.info["CellsNumberPrior"][0],self.info["CellsNumberPrior"][1],])
        distance_hp_corner = tensor([self.info["PositionHPPrior"][1], self.info["PositionHPPrior"][0]-2])
        hp_boxes = []
        pos_hps = stack(list(where(material_ids == max(material_ids))), dim=0).T

        for idx in range(len(pos_hps)):
            try:
                pos_hp = pos_hps[idx]
                corner_ll, corner_ur = get_box_corners(pos_hp, size_hp_box, distance_hp_corner, self.inputs.shape[1:], run_name=self.file_name,)
                tmp_input = self.inputs[:, corner_ll[0] : corner_ur[0], corner_ll[1] : corner_ur[1]].detach().clone()
                tmp_label = self.label[:, corner_ll[0] : corner_ur[0], corner_ll[1] : corner_ur[1]].detach().clone()

                tmp_mat_ids = stack(list(where(tmp_input == max(material_ids))), dim=0).T
                if len(tmp_mat_ids) > 1:
                    for i in range(len(tmp_mat_ids)):
                        tmp_pos = tmp_mat_ids[i]
                        if (tmp_pos[1:2] != distance_hp_corner).all():
                            tmp_input[tmp_pos[0], tmp_pos[1], tmp_pos[2]] = 0

                tmp_hp = HeatPump(id=idx, pos=pos_hp, orientation=0, inputs=tmp_input, dist_corner_hp=distance_hp_corner, label=tmp_label, device=device,)
                if "SDF" in self.info["Inputs"]:
                    tmp_hp.recalc_sdf(self.info)

                hp_boxes.append(tmp_hp)
                logging.info(
                    f"HP BOX at {pos_hp} is with ({corner_ll}, {corner_ur}) in domain"
                )
            except:
                logging.warning(f"BOX of HP {idx} at {pos_hp} is not in domain")
                
        return hp_boxes

    def add_hp(self, hp: "HeatPump", prediction_field: tensor):
        # compose learned fields into large domain with list of ids, pos, orientations
        for i in range(prediction_field.shape[0]):
            for j in range(prediction_field.shape[1]):
                x, y = self.coord_trafo(
                    hp.pos,
                    (i - hp.dist_corner_hp[0], j - hp.dist_corner_hp[1]),
                    hp.orientation,
                )
                if (
                    0 <= x < self.prediction.shape[0]
                    and 0 <= y < self.prediction.shape[1]
                ):
                    self.prediction[x, y] = self.stitching(
                        self.prediction[x, y], prediction_field[i, j]
                    ) # TODO TODO changed stitching to pointwise max (np.maximum instead of max) - does this break now!
                    # TODO TODO or problem with np vs torch?

    def coord_trafo(self, fixpoint: tuple, position: tuple, orientation: float):
        """
        transform coordinates from domain to hp
        """
        x = (
            fixpoint[0]
            + int(position[0] * cos(orientation))
            + int(position[1] * sin(orientation))
        )
        y = (
            fixpoint[1]
            + int(position[0] * sin(orientation))
            + int(position[1] * cos(orientation))
        )
        return x, y

    def plot(self, fields: str = "t", folder: str = "", name: str = "test"):
        properties = expand_property_names(fields)
        n_subplots = len(properties)
        if "t" in fields:
            n_subplots += 2
        plt.subplots(n_subplots, 1, sharex=True, figsize=(20, 3 * (n_subplots)))
        idx = 1
        for property in properties:
            plt.subplot(n_subplots, 1, idx)
            if property == "Temperature [C]":
                plt.imshow(self.prediction.T)
                plt.gca().invert_yaxis()
                plt.xlabel("x [cells]")
                plt.ylabel("y [cells]")
                _aligned_colorbar(label=f"Predicted {property}")
                idx += 1
                plt.subplot(n_subplots, 1, idx)
                if self.normed:
                    self.label = self.reverse_norm(self.label, property)
                    self.normed = False
                plt.imshow(abs(self.prediction.T - squeeze(self.label.T)))
                plt.gca().invert_yaxis()
                plt.xlabel("x [cells]")
                plt.ylabel("y [cells]")
                _aligned_colorbar(label=f"Absolute error in {property}")
                idx += 1
                plt.subplot(n_subplots, 1, idx)
                plt.imshow(self.label.T)
            elif property == "Original Temperature [C]":
                field = self.prediction_2HP
                property = "1st Prediction of Temperature [C]"
                plt.imshow(field.T)
            else:
                field = self.get_input_field_from_name(property)
                field = self.reverse_norm(field, property)
                plt.imshow(field.T)
            plt.gca().invert_yaxis()
            plt.xlabel("x [cells]")
            plt.ylabel("y [cells]")
            _aligned_colorbar(label=property)
            idx += 1
        plt.savefig(f"{folder}/{name}.pgf", format="pgf")


def get_box_corners(pos_hp, size_hp_box, distance_hp_corner, domain_shape, run_name: str = "unknown"):
    corner_ll = (pos_hp - distance_hp_corner).to(dtype=torch_long)  # corner lower left
    corner_ur = (pos_hp + size_hp_box - distance_hp_corner).to(dtype=torch_long) # corner upper right
    # if corner_ll[0] < 0 or corner_ur[0] >= domain_shape[0] or corner_ll[1] < 0 or corner_ur[1] >= domain_shape[1]:
    #     # move file from "Inputs" to "broken/Inputs"
    #     logging.warning(f"HP BOX at {pos_hp} is with x=({corner_ll[0]}, {corner_ur[0]}) in x-direction (0, {domain_shape[0]}) or y=({corner_ll[1]}, {corner_ur[1]}) in y-direction (0, {domain_shape[1]}) not in domain for {run_name}")
    #     origin_2hp_prep = "/home/pelzerja/Development/datasets_prepared/2hps_demonstrator/dataset_2hps_1fixed_1000dp_grad_p"
    #     # TODO rm absolute path
    #     shutil.move(os.path.join(origin_2hp_prep, "Inputs", run_name), os.path.join(origin_2hp_prep, "broken", "Inputs", f"{run_name.split('.')[0]}_hp_pos_outside_domain.pt"))
    #     shutil.move(os.path.join(origin_2hp_prep, "Labels", run_name), os.path.join(origin_2hp_prep, "broken", "Labels", run_name))
    assert (corner_ll[0] >= 0 and corner_ur[0] < domain_shape[0]), f"HP BOX at {pos_hp} is with x=({corner_ll[0]}, {corner_ur[0]}) in x-direction (0, {domain_shape[0]}) not in domain for {run_name}"
    assert (corner_ll[1] >= 0 and corner_ur[1] < domain_shape[1]), f"HP BOX at {pos_hp} is with y=({corner_ll[1]}, {corner_ur[1]}) in y-direction (0, {domain_shape[1]}) not in domain for {run_name}"

    return corner_ll, corner_ur