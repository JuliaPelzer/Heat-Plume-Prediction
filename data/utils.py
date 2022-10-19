import pickle
import os
from typing import List, Dict, Tuple
import numpy as np
import torch


def save_pickle(data_dict, file_name):
    """Save given data dict to pickle file file_name in models/

    Parameters
    ----------
    data_dict : e.g. {"dataset": dataset,
        "cifar_mean": cifar_mean,
        "cifar_std": cifar_std}
    file_name : str
        Name of the file to be saved, ends with .p
    """
    directory = 'models'
    if not os.path.exists(directory):
        os.makedirs(directory)
    pickle.dump(data_dict, open(os.path.join(directory, file_name), 'wb', 5))


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


class PhysicalVariable:
    def __init__(self, name: str, value: torch.DoubleTensor = None):  # TODO ? default value + type
        self.id_name = name
        self.name_without_unit, self.unit = separate_property_unit(name)
        self.value = value
        # TODO required to put mean, std somewhere else? (other level / class)
        self.mean_orig: float = None
        self.std_orig: float = None

    def __repr__(self):
        return f"{self.name_without_unit} (in {self.unit})"

    def __len__(self) -> int:
        # assert np.size(self.value) != 1, "value not set"
        return np.size(self.value)

    def shape(self) -> Tuple[int]:
        # assert np.size(self.value) != 1, "value not set"
        return self.value.shape

    def __eq__(self, o) -> bool:
        if not isinstance(o, PhysicalVariable):
            return False
        return self.id_name == o.id_name and np.array_equal(self.value, o.value)

    def calc_mean(self):
        # TODO requires Keepdim=True and dim=(1,2,3) ???

        # check if type is correct to calc mean (not int!)
        if self.value.type != torch.DoubleTensor:
            self.value = self.value.type(torch.DoubleTensor)
        try:
            self.mean_orig = torch.mean(self.value)
        except:
            self.mean_orig = np.mean(self.value)

    def calc_std(self):
        # check if type is correct to calc mean (not int!)
        if self.value.type != torch.DoubleTensor:
            self.value = self.value.type(torch.DoubleTensor)
        try:
            self.std_orig = torch.std(self.value)
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


def test_physical_variable():
    time = "now"
    temp = PhysicalVariables(time)
    temp["temperature"] = 1
    print("value", temp["temperature"].value)
    temp["temperature"] = 2
    print("repr", temp["temperature"])
    print("name", temp["temperature"].name_without_unit)
    temp["temperature"] = 4
    print("value", temp["temperature"].value)
    print("size", temp["temperature"].__sizeof__())
    temp["Pressure [Pa]"] = 3
    print("value pressure", temp["Pressure [Pa]"].value)
    print("names", temp.get_names_without_unit())
    print("ids", temp.get_ids_list())
    print("repr", temp["Pressure [Pa]"])
    print(temp.keys(), temp.values())

    time = "now [s]"
    expected_temperature = PhysicalVariable("Temperature [K]")
    properties = {"temperature": PhysicalVariable("Temperature [K]"),
                  "pressure": PhysicalVariable("Pressure [Pa]")}
    physical_properties = PhysicalVariables(time, properties)
    physical_properties["pressure"] = 2
    physical_properties["temperature"] = 3
    physical_properties["ID [-]"] = 0

    print(list(physical_properties.keys()) == [
          'temperature', 'pressure', 'ID [-]'], physical_properties.values())


if __name__ == "__main__":
    test_physical_variable()
