import os
from typing import List, Dict, Tuple
import numpy as np
import yaml
from torch import Tensor, DoubleTensor, equal, mean, std
from dataclasses import dataclass, field

class PhysicalVariable:
    def __init__(self, name: str, value: DoubleTensor = None):
        self.id_name = name
        self.name_without_unit, self.unit = separate_property_unit(name)
        self.value = value
        self.mean_orig: float = None
        self.std_orig: float = None

    def __repr__(self):
        return f"{self.name_without_unit} (in {self.unit}) with {self.shape()} elements"

    def dim(self) -> int:
        return len(self.value.shape)

    def shape(self) -> Tuple[int]:
        try:
            return tuple(self.value.shape)
        except Exception as e:
            if np.size(self.value) == 1:
                return np.size(self.value)

    def __eq__(self, o) -> bool:
        if not isinstance(o, PhysicalVariable):
            return False
        try:
            return self.id_name == o.id_name and equal(self.value, o.value)
        except:
            return self.id_name == o.id_name and self.value == o.value

    def calc_mean(self):
        # TODO requires Keepdim=True and dim=(1,2,3) ???

        # check if type is correct to calc mean (not int!)
        if self.value.type != DoubleTensor:
            self.value = self.value.type(DoubleTensor)
        try:
            self.mean_orig = mean(self.value)
        except:
            self.mean_orig = np.mean(self.value)

    def calc_std(self):
        # check if type is correct to calc mean (not int!)
        if self.value.type != DoubleTensor:
            self.value = self.value.type(DoubleTensor)
        try:
            self.std_orig = std(self.value)
        except:
            self.std_orig = np.std(self.value)

class PhysicalVariables(dict):
    def __init__(self, time: str, properties: List[str] = None):
        super().__init__()
        if properties is None:
            properties = []
        for prop in properties:
            self[prop] = PhysicalVariable(prop)
        self.time = time

    def __setitem__(self, key: str, value: np.ndarray):
        if key not in self.keys():
            super().__setitem__(key, PhysicalVariable(key, value))
        self[key].value = value

    def get_names_without_unit(self):
        return [var.name_without_unit for _, var in self.items()]

    def get_ids_list(self):
        return [var.id_name for _, var in self.items()]

    def get_number_of_variables(self):
        return len(self.keys())

@dataclass
class DataPoint():
    run_id: int
    inputs: PhysicalVariables = None
    labels: PhysicalVariables = None

    def __repr__(self) -> str:
        return f"DataPoint at {self.run_id} with {self.len_inputs()} input variables and {self.len_labels()} output variables"
    
    def len_inputs(self) -> np.ndarray:
        return self.inputs.get_number_of_variables()

    def len_labels(self) -> np.ndarray:
        return self.labels.get_number_of_variables()

    def get_loc_hp(self):
        try: #TODO problematic with SDF?
            ids = self.inputs["Material ID"].value
        except:
            ids = self.inputs["SDF"].value
        max_id = ids.max()
        loc_hp = np.array(np.where(ids == max_id)).squeeze()
        return loc_hp
        
@dataclass
class Batch():
    batch_id: int
    inputs: Tensor = field(default_factory=Tensor) # init as empty tensor
    labels: Tensor = field(default_factory=Tensor)
    # inputs, labels: in the beginning C,H,W,(D), later: Run_ID/Datapoint,C,H,W,(D)

    def dim(self):
        return len(Tensor.size(self.inputs))
    
    def __len__(self):
        return Tensor.size(self.inputs)[0]

def separate_property_unit(property_in: str) -> List[str]:
    """Separate property and unit in input string"""
    index_open = property_in.find(' [')
    index_close = property_in.find(']')

    assert index_open == -1 and index_close == -1 or index_open != - \
        1 and index_close != - \
        1, "input string has to contain both '[' and ']' or neither"
    if index_open != -1 and index_close != -1:
        name = property_in[:index_open]
        unit = property_in[index_open+2:index_close]
    else:
        name = property_in
        unit = None

    return name, unit

def get_dimensions(path:str) -> Tuple[int, int, int]:
    # read json file for dimensions
    with open(f"{path}/inputs/settings.yaml", "r") as f:
        perm_settings = yaml.safe_load(f)
    dimensions_of_datapoint = perm_settings["grid"]["ncells"]
    return dimensions_of_datapoint

def load_settings(path:str, file_name="settings") -> Dict:
	with open(f"{path}/{file_name}.yaml", "r") as file:
		settings = yaml.load(file, Loader=yaml.SafeLoader)
	return settings

def save_settings(settings:Dict, path:str, name_file:str="settings"):
	with open(f"{path}/{name_file}.yaml", "w") as file:
		yaml.dump(settings, file)

@dataclass
class SettingsTraining:
    dataset_name: str
    device: str
    epochs: int
    model_choice: str
    inputs: str
    finetune: bool
    path_to_model: str
    name_folder_destination: str
    path_to_datasets: str = "/home/pelzerja/Development/simulation_groundtruth_pflotran/Phd_simulation_groundtruth/datasets"

    def __post_init__(self):
        self.path_to_model = os.path.join("runs", self.path_to_model)
    
    def save(self):
        save_settings(self.__dict__, os.path.join("runs", self.name_folder_destination), "settings_training")

def _assertion_error_2d(datapoint:DataPoint):
    # TODO how/where to test whether reduce_to_2D worked?
    """
    checks if the data is 2D or 3D - else: error
    """
    for input_var in datapoint.inputs.values():
        assert input_var.dim() == 2 or input_var.dim() == 3, "Input data is neither 2D nor 3D but {}D".format(input_var.dim())
        break
    for output_var in datapoint.labels.values():
        assert output_var.dim() == 2 or output_var.dim() == 3, "Input data is neither 2D nor 3D but {}D".format(output_var.dim())
        break

def _test_physical_variable():
    time = "now [s]"
    expected_temperature = PhysicalVariable("Temperature [K]")
    properties = ["Temperature [K]", "Pressure [Pa]"]
    physical_properties = PhysicalVariables(time, properties)
    physical_properties["Temperature [K]"]=Tensor([3])
    physical_properties["Pressure [Pa]"]=2
    physical_properties["ID [-]"]=0
    print(physical_properties["Temperature [K]"])
    assert physical_properties["Temperature [K]"].__repr__()=="Temperature (in K) with (1,) elements", "repr not working"
    assert physical_properties["Pressure [Pa]"].__repr__()=="Pressure (in Pa) with 1 elements", "repr not working"
    assert physical_properties.get_names_without_unit()==["Temperature", "Pressure", "ID"], "get_names_without_unit() not working"
    assert physical_properties["Temperature [K]"].value == 3, "value not set correctly"
    assert len(physical_properties)==3, "len not working"
    assert physical_properties["Temperature [K]"].unit == "K", "unit not set correctly"
    assert physical_properties["ID [-]"].value == 0, "value not set correctly"
    assert physical_properties["ID [-]"].unit == "-", "unit not set correctly"
    assert physical_properties["ID [-]"].__repr__()=="ID (in -) with 1 elements", "repr not working"
    assert physical_properties["ID [-]"].name_without_unit == "ID", "name_without_unit not set correctly"
    assert physical_properties["ID [-]"].id_name == "ID [-]", "id_name not set correctly"
    assert physical_properties.get_ids_list()==["Temperature [K]", "Pressure [Pa]", "ID [-]"], "get_ids not working"
    assert list(physical_properties.keys()) == ["Temperature [K]", "Pressure [Pa]", "ID [-]"], "keys not working"
    # test PhysicalVariable.__eq__()
    assert physical_properties["Temperature [K]"] != expected_temperature, "PhysicalVariable.__eq__() failed"
    expected_temperature.value = 3
    assert expected_temperature.value == 3, "value not set correctly"
    assert physical_properties["Temperature [K]"] == expected_temperature, "PhysicalVariable.__eq__() failed"

if __name__ == "__main__":
    _test_physical_variable()
