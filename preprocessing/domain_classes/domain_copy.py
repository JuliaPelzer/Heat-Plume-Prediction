from copy import deepcopy
import logging
import os
import sys
from math import cos, sin
from tqdm.auto import tqdm
import yaml
from torch import long as torch_long
from torch import max, ones, stack, tensor, where, cat, load

sys.path.append("/home/pelzerja/pelzerja/test_nn/1HP_NN")  # relevant for remote
sys.path.append("/home/pelzerja/Development/1HP_NN")  # relevant for local
from preprocessing.domain_classes.heat_pump_copy import HeatPumpBox
from preprocessing.domain_classes.stitching import Stitching
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
        pos_hps = stack(list(where(material_ids == max(material_ids))), dim=0).T
        names_inputs = [self.get_name_from_index(i) for i in range(self.inputs.shape[0])]

        for idx in tqdm(range(len(pos_hps)), desc="Extracting HP-Boxes"):
            try:
                pos_hp = pos_hps[idx]
                corner_ll = get_box_corners(pos_hp, size_hp_box, distance_hp_corner, self.inputs.shape[1:], run_name=self.file_name,)
                tmp_input = self.inputs[:, corner_ll[0] : corner_ll[0] + size_hp_box[0], corner_ll[1] : corner_ll[1] + size_hp_box[1]].detach().clone()
                tmp_label = self.label[:, corner_ll[0] : corner_ll[0] + size_hp_box[0], corner_ll[1] : corner_ll[1] + size_hp_box[1]].detach().clone()

                tmp_mat_ids = stack(list(where(tmp_input == max(material_ids))), dim=0).T
                if len(tmp_mat_ids) > 1:
                    for i in range(len(tmp_mat_ids)):
                        tmp_pos = tmp_mat_ids[i]
                        if (tmp_pos[1:2] != distance_hp_corner).all():
                            tmp_input[tmp_pos[0], tmp_pos[1], tmp_pos[2]] = 0

                tmp_hp = HeatPumpBox(id=idx, pos=pos_hp, orientation=0, inputs=tmp_input, names=names_inputs, dist_corner_hp=distance_hp_corner, label=tmp_label, device=device,)
                if "SDF" in self.info["Inputs"]:
                    tmp_hp.recalc_sdf(self.info)

                hp_boxes.append(tmp_hp)
                logging.info(f"HP BOX at {pos_hp} is in domain, starting in corner {corner_ll}")
            except:
                logging.warning(f"BOX of HP {idx} at {pos_hp} is not in domain")
                
        return hp_boxes

    def extract_ep_box(self, hp: "HeatPumpBox", params: dict, device:str = "cpu"):
        # inspired by ep.assemble_inputs + extract_hp_boxes

        # TODO check achtung orientierung hp.primary_temp_field.shape for hp-size
        corner_ll = get_box_corners(hp.pos, tensor(hp.primary_temp_field.shape), hp.dist_corner_hp, self.inputs.shape[1:], run_name=self.file_name,)
        # get "inputs" there from domain (only g,k (?) not s, i)
        start_curr = params["start_curr_box"] + corner_ll[0]
        # TODO inputs HARDCODED to "gk" from "gksi"
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

        return x, y

def get_box_corners(pos_hp, size_hp_box, distance_hp_corner, domain_shape, run_name: str = "unknown"):
    corner_ll = (pos_hp - distance_hp_corner) # corner lower left
    
    try:
        corner_ll = corner_ll.to(dtype=torch_long) 
    except: ...

    assert (corner_ll[0] >= 0 and (corner_ll + size_hp_box)[0] < domain_shape[0]), f"HP BOX at {pos_hp} is with x=({corner_ll[0]}, {(corner_ll + size_hp_box)[0]}) in x-direction (0, {domain_shape[0]}) not in domain for {run_name}"
    assert (corner_ll[1] >= 0 and (corner_ll + size_hp_box)[1] < domain_shape[1]), f"HP BOX at {pos_hp} is with y=({corner_ll[1]}, {(corner_ll + size_hp_box)[1]}) in y-direction (0, {domain_shape[1]}) not in domain for {run_name}"

    return corner_ll