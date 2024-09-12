from copy import deepcopy
import logging
import os
import sys
from math import cos, sin
from tqdm.auto import tqdm
import yaml
import numpy as np
from torch import long as torch_long
from torch import max, maximum, ones, zeros, stack, tensor, where, cat, load, FloatTensor

sys.path.append("/home/pelzerja/pelzerja/test_nn/1HP_NN")  # relevant for remote
sys.path.append("/home/pelzerja/Development/1HP_NN")  # relevant for local
from preprocessing.domain_classes.heat_pump import HeatPumpBox
from preprocessing.domain_classes.stitching import Stitching
from preprocessing.transforms import SignedDistanceTransform
import utils.utils_args as ua



class Domain:
    def __init__(
        self, info_path: str, stitching_method: str = "max", file_name: str = "RUN_0.pt", device = "cpu"):
        self.info = ua.load_yaml(info_path/"info.yaml", Loader=yaml.FullLoader)
        self.background_temperature: float = 10.6
        self.inputs: tensor = self.load_datapoint(info_path, case="Inputs", file_name=file_name)
        self.label: tensor = self.load_datapoint(info_path, case="Labels", file_name=file_name)
        size: tuple[int, int] = [self.info["CellsNumber"][0], self.info["CellsNumber"][1], ]  # (x, y), cell-ids
        self.prediction: tensor = (ones(size) * self.background_temperature).to(device)
        self.stitching: Stitching = Stitching(stitching_method, self.background_temperature)
        self.file_name: str = file_name

    def load_datapoint(self, dataset_domain_path: str, case: str = "Inputs", file_name="RUN_0.pt"):
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
            out_min, out_max = (0, 1)  # TODO Achtung! Hardcoded, values same as in transforms.NormalizeTransform.out_min/max
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
            out_min, out_max = (0,1,)  # TODO Achtung! Hardcoded, values same as in transforms.NormalizeTransform.out_min/max
            delta = max_val - min_val
            data = (data - out_min) / (out_max - out_min) * delta + min_val
        elif norm_fct == "Standardize":
            data = data * std_val + mean_val
        elif norm_fct is None:
            pass
        else:
            raise ValueError(f"Normalization type '{self.norm_fct['Norm']}' not recognized")
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
        distance_hp_corner = tensor([self.info["PositionHPPrior"][1], self.info["PositionHPPrior"][0]])
        hp_boxes = []
        hp_inputs = []
        pos_hps = stack(list(where(material_ids == max(material_ids))), dim=0).T
        names_inputs = [self.get_name_from_index(i) for i in range(self.inputs.shape[0])]

        dummy_mat_id_field = zeros(size_hp_box.tolist())
        dummy_mat_id_field[distance_hp_corner[0], distance_hp_corner[1]] = 1
        
        for idx in tqdm(range(len(pos_hps)), desc="Extracting HP-Boxes"):
            try:
                pos_hp = pos_hps[idx]
                corner_ll = get_box_corners(pos_hp, distance_hp_corner)
                check_box_corners(corner_ll, size_hp_box, self.inputs.shape[1:], pos_hp, run_name=self.file_name)
                tmp_input = self.inputs[:, corner_ll[0] : corner_ll[0] + size_hp_box[0], corner_ll[1] : corner_ll[1] + size_hp_box[1]].detach().clone()
                tmp_label = self.label[:, corner_ll[0] : corner_ll[0] + size_hp_box[0], corner_ll[1] : corner_ll[1] + size_hp_box[1]].detach().clone()

                # if "i" in inputs: make sure to cut out none-primary hps
                if "Material ID" in self.info["Inputs"]:
                    idx_i = self.get_index_from_name("Material ID")
                    tmp_input[idx_i] = dummy_mat_id_field # Attention: no deepcopy

                tmp_hp = HeatPumpBox(id=idx, pos=pos_hp, orientation=0, inputs=tmp_input, names=names_inputs, dist_corner_hp=distance_hp_corner, label=tmp_label, device=device,)
                if "SDF" in self.info["Inputs"]:
                    tmp_hp.recalc_sdf(self.info)

                hp_boxes.append(tmp_hp)
                hp_inputs.append(np.array(tmp_input))
                logging.info(f"HP BOX at {pos_hp} is in domain, starting in corner {corner_ll}")
            except:
                logging.warning(f"BOX of HP {idx} at {pos_hp} is not in domain")

        return hp_boxes, FloatTensor(np.array(hp_inputs))

    def extract_ep_box(self, hp: "HeatPumpBox", params: dict, device:str = "cpu"):
        # inspired by ep.assemble_inputs + extract_hp_boxes

        # TODO check achtung orientierung hp.primary_temp_field.shape for hp-size
        corner_ll = get_box_corners(hp.pos, hp.dist_corner_hp)
        check_box_corners(corner_ll, tensor(hp.primary_temp_field.shape), self.inputs.shape[1:], hp.pos, run_name=self.file_name)
        # get "inputs" there from domain (only g,k (?) not s, i)
        start_curr = params["start_curr_box"] + corner_ll[0]
        # TODO inputs HARDCODED to "gk" from "gksi"
        assert self.inputs.shape[0] == 4, "Only implemented for gk inputs"
        input_curr = self.inputs[:2, start_curr : start_curr + params["box_size"], corner_ll[1] : (corner_ll + tensor(hp.primary_temp_field.shape))[1]].to(device)

        # combine with second half of hp.primary_temp_field as inputs for extend plumes
        start_prior = params["start_prior_box"]
        input_prior_T = hp.primary_temp_field[start_prior : start_prior + params["box_size"],:].detach() # before: only 2D
        # print(start_prior, params["box_size"], hp.primary_temp_field.shape, corner_ll[0])
        if len(input_prior_T.shape) == 2:
            input_prior_T = input_prior_T.unsqueeze(0)
        # print(input_curr.shape, input_prior_T.shape)
        input_all = cat((input_curr, input_prior_T), dim=0).unsqueeze(0)

        return input_all, corner_ll

    def add_hp(self, hp: "HeatPumpBox"):
        '''
        Stitching of normed HP-Boxes into domain's unnormed 'prediction' field
        '''
        prediction_field = self.reverse_norm(deepcopy(hp.primary_temp_field), property="Temperature [C]") # for adding to domain
        # compose learned fields into large domain with list of ids, pos, orientations
        for i in range(prediction_field.shape[0]):
            for j in range(prediction_field.shape[1]):
                x, y = self.coord_trafo(hp.pos, (i - hp.dist_corner_hp[0], j - hp.dist_corner_hp[1]), hp.orientation,)
                if (0 <= x < self.prediction.shape[0] and 0 <= y < self.prediction.shape[1]):
                    self.prediction[x, y] = self.stitching(self.prediction[x, y], prediction_field[i, j])

    def coord_trafo(self, fixpoint: tuple, position: tuple, orientation: float):
        """
        transform coordinates from domain to hp
        """
        x = (fixpoint[0] + int(position[0] * cos(orientation)) + int(position[1] * sin(orientation)))
        y = (fixpoint[1] + int(position[0] * sin(orientation)) + int(position[1] * cos(orientation)))

        return int(x), int(y) # Attention: int() for indexing
class DomainBatch(Domain):
    def __init__(self, info_path: str, stitching_method: str = "max", file_name: str = "RUN_0.pt", device="cpu"):
        super().__init__(info_path, stitching_method, file_name, device)
        assert stitching_method == "max", "Only 'max' stitching implemented for Domain2"

    def extract_hp_boxes(self, device:str) -> list:
        # TODO decide: get hp_boxes based on grad_p or based on v or get squared boxes around hp
        material_ids = self.get_input_field_from_name("Material ID")
        size_hp_box = tensor([self.info["CellsNumberPrior"][0],self.info["CellsNumberPrior"][1],])
        distance_hp_corner = tensor([self.info["PositionHPPrior"][1], self.info["PositionHPPrior"][0]])
        hp_inputs = []
        hp_labels = []
        hp_poss = []

        dummy_mat_id_field = calc_mat_id_field(size_hp_box, distance_hp_corner)
        dummy_sdf = calc_sdf(dummy_mat_id_field, dist_corner_hp=distance_hp_corner)
        
        tqdm_pos_hps = stack(list(where(material_ids == max(material_ids))), dim=0).T
        for idx in tqdm(range(len(tqdm_pos_hps)), desc="Extracting HP-Boxes"):
            try:
                pos_hp = tqdm_pos_hps[idx]
                corner_ll = get_box_corners(tqdm_pos_hps[idx], distance_hp_corner)
                check_box_corners(corner_ll, size_hp_box, self.inputs.shape[1:], tqdm_pos_hps[idx], run_name=self.file_name)
                tmp_input = self.inputs[:, corner_ll[0] : corner_ll[0] + size_hp_box[0], corner_ll[1] : corner_ll[1] + size_hp_box[1]].detach().clone()
                tmp_label = self.label[:, corner_ll[0] : corner_ll[0] + size_hp_box[0], corner_ll[1] : corner_ll[1] + size_hp_box[1]].detach().clone()

                # if "i" in inputs: make sure to cut out none-primary hps
                if "Material ID" in self.info["Inputs"]:
                    idx_i = self.get_index_from_name("Material ID")
                    tmp_input[idx_i] = dummy_mat_id_field # Attention: no deepcopy
                if "SDF" in self.info["Inputs"]:
                    idx_s = self.get_index_from_name("SDF")
                    tmp_input[idx_s] = dummy_sdf # Attention: no deepcopy

                hp_inputs.append(np.array(tmp_input))
                hp_labels.append(np.array(tmp_label))
                hp_poss.append(np.array(pos_hp))
                logging.info(f"HP BOX at {tqdm_pos_hps[idx]} is in domain, starting in corner {corner_ll}")
            except:
                logging.warning(f"BOX of HP {idx} at {tqdm_pos_hps[idx]} is not in domain")

        return {"pos": FloatTensor(np.array(hp_poss)).to(device), "inputs": FloatTensor(np.array(hp_inputs)).to(device), "labels": FloatTensor(np.array(hp_labels)).to(device)}, distance_hp_corner.to(device)

    def add_predictions(self, hps:dict, distance_hp_corner:tensor):
        predictions = self.reverse_norm(hps["predictions"], property="Temperature [C]") # for adding to domain

        for pos, prediction_field in tqdm(zip(hps["pos"],predictions), desc="Adding box-predictions (+ep) to domain"):
            x_min, y_min = self.coord_trafo(pos, (0 - distance_hp_corner[0], 0 - distance_hp_corner[1]), orientation=0)
            x_max, y_max = self.coord_trafo(pos, (prediction_field.shape[-2] - distance_hp_corner[0], prediction_field.shape[-1] - distance_hp_corner[1]), orientation=0)
            
            if (0 <= x_min < self.prediction.shape[-2] and 0 <= y_min < self.prediction.shape[-1] and 0 <= x_max < self.prediction.shape[-2] and 0 <= y_max < self.prediction.shape[-1]):
                self.prediction[x_min:x_max, y_min:y_max] = maximum(self.prediction[x_min:x_max, y_min:y_max], prediction_field)

def calc_mat_id_field(size_hp_box, distance_hp_corner):
    dummy_mat_id_field = zeros(size_hp_box.tolist())
    dummy_mat_id_field[distance_hp_corner[0], distance_hp_corner[1]] = 1
    return dummy_mat_id_field

def calc_sdf(input_mat_id, dist_corner_hp):
    return SignedDistanceTransform().sdf(input_mat_id.detach().clone(), dist_corner_hp)

def get_box_corners(pos_hp, distance_hp_corner) -> tensor:
    corner_ll = (pos_hp - distance_hp_corner) # corner lower left
    
    try:
        corner_ll = corner_ll.to(dtype=torch_long) 
    except: ...

    return corner_ll

def check_box_corners(corner_ll: tensor, size_hp_box: tensor, domain_shape: tuple[int, int], pos_hp: int, run_name: str = "unknown"):
    
    assert (corner_ll[0] >= 0 and (corner_ll + size_hp_box)[0] < domain_shape[0]), f"HP BOX at {pos_hp} is with x=({corner_ll[0]}, {(corner_ll + size_hp_box)[0]}) in x-direction (0, {domain_shape[0]}) not in domain for {run_name}"
    assert (corner_ll[1] >= 0 and (corner_ll + size_hp_box)[1] < domain_shape[1]), f"HP BOX at {pos_hp} is with y=({corner_ll[1]}, {(corner_ll + size_hp_box)[1]}) in y-direction (0, {domain_shape[1]}) not in domain for {run_name}"