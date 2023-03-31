import os
from typing import Dict
import yaml
from torch import Tensor
from dataclasses import dataclass
import pathlib


def load_settings(path: str, file_name="settings") -> Dict:
    path = pathlib.Path(path)
    with open(path.joinpath(f"{file_name}.yaml"), "r") as file:
        settings = yaml.load(file)
    return settings


def save_settings(settings: Dict, path: str, name_file: str = "settings"):
    path = pathlib.Path(path)
    with open(path.joinpath(f"{name_file}.yaml"), "w") as file:
        yaml.dump(settings, file)


@dataclass
class SettingsTraining:
    dataset_name: str
    device: str
    epochs: int
    model_choice: str
    finetune: bool
    path_to_model: str
    name_folder_destination: str
    datasets_path: str = "/home/pelzerja/Development/simulation_groundtruth_pflotran/Phd_simulation_groundtruth/datasets"

    def __post_init__(self):
        self.path_to_model = os.path.join("runs", self.path_to_model)

    def save(self):
        save_settings(self.__dict__, os.path.join(
            "runs", self.name_folder_destination), "settings_training")


def _assertion_error_2d(datapoint):
    # TODO how/where to test whether reduce_to_2D worked?
    # TODO use/ fix this?
    """
    checks if the data is 2D or 3D - else: error
    """
    for input_var in datapoint.inputs.values():
        assert input_var.dim() == 2 or input_var.dim(
        ) == 3, "Input data is neither 2D nor 3D but {}D".format(input_var.dim())
        break
    for output_var in datapoint.labels.values():
        assert output_var.dim() == 2 or output_var.dim(
        ) == 3, "Input data is neither 2D nor 3D but {}D".format(output_var.dim())
        break


def _test_physical_variable():
    # TODO test is cringe, remove this
    time = "now [s]"
    expected_temperature = PhysicalVariable("Temperature [K]")
    properties = ["Temperature [K]", "Pressure [Pa]"]
    physical_properties = PhysicalVariables(time, properties)
    physical_properties["Temperature [K]"] = Tensor([3])
    physical_properties["Pressure [Pa]"] = 2
    physical_properties["ID [-]"] = 0
    print(physical_properties["Temperature [K]"])
    assert physical_properties["Temperature [K]"].__repr__(
    ) == "Temperature (in K) with (1,) elements", "repr not working"
    assert physical_properties["Pressure [Pa]"].__repr__(
    ) == "Pressure (in Pa) with 1 elements", "repr not working"
    assert physical_properties.get_names_without_unit(
    ) == ["Temperature", "Pressure", "ID"], "get_names_without_unit() not working"
    assert physical_properties["Temperature [K]"].value == 3, "value not set correctly"
    assert len(physical_properties) == 3, "len not working"
    assert physical_properties["Temperature [K]"].unit == "K", "unit not set correctly"
    assert physical_properties["ID [-]"].value == 0, "value not set correctly"
    assert physical_properties["ID [-]"].unit == "-", "unit not set correctly"
    assert physical_properties["ID [-]"].__repr__(
    ) == "ID (in -) with 1 elements", "repr not working"
    assert physical_properties["ID [-]"].name_without_unit == "ID", "name_without_unit not set correctly"
    assert physical_properties["ID [-]"].id_name == "ID [-]", "id_name not set correctly"
    assert physical_properties.get_ids_list(
    ) == ["Temperature [K]", "Pressure [Pa]", "ID [-]"], "get_ids not working"
    assert list(physical_properties.keys()) == [
        "Temperature [K]", "Pressure [Pa]", "ID [-]"], "keys not working"
    # test PhysicalVariable.__eq__()
    assert physical_properties["Temperature [K]"] != expected_temperature, "PhysicalVariable.__eq__() failed"
    expected_temperature.value = 3
    assert expected_temperature.value == 3, "value not set correctly"
    assert physical_properties["Temperature [K]"] == expected_temperature, "PhysicalVariable.__eq__() failed"


if __name__ == "__main__":
    _test_physical_variable()
