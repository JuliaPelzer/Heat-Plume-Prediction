import pickle
import os
from typing import List, Dict
import numpy as np

# from dataclasses import dataclass

# @dataclass
# class Location():
#     name:str
#     path:str=None

#     def check_path(self):
#         assert os.path.exists(self.path), f"Path {self.path} does not exist"
# TODO later


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


def separate_property_unit(property_in:str) -> List[str]:
    """Separate property and unit in input string"""
    index_open = property_in.find(' [')
    index_close = property_in.find(']')
    
    assert index_open == -1 and index_close == -1 or index_open != -1 and index_close != -1, "input string has to contain both '[' and ']' or neither"
    if index_open != -1 and index_close != -1:
        name = property_in[:index_open]
        unit = property_in[index_open+2:index_close]
    else:
        name = property_in
        unit = None

    return name, unit


class PhysicalVariable:
    def __init__(self, name:str): #TODO ? default value + type
        self.id_name = name
        self.name_without_unit, self.unit = separate_property_unit(name)
        self.value = None

    def __repr__(self):
        return f"{self.name_without_unit} (in {self.unit})"

    # def fill(self, value):
    #     # todo assert input type in certain shape
    #     self.value = value

    def __sizeof__(self) -> int:
        return np.size(self.value)
    
class PhysicalVariables(dict):
    def __init__(self, time:str, properties:Dict[str, PhysicalVariable]=[]):
        super().__init__(properties)
        self.time = time

    def __setitem__(self, key:str, value:PhysicalVariable):
        self[key].value = value

    def get_names_without_unit(self):
        return [var.name_without_unit for _, var in self.items()]

    def get_ids(self):
        return [var.id_name for _, var in self.items()]