import os
import sys
from typing import List

sys.path.append("/home/pelzerja/pelzerja/test_nn/1HP_NN")  # relevant for remote
sys.path.append("/home/pelzerja/Development/1HP_NN")  # relevant for local
from data_stuff.utils import save_yaml


def save_config_of_separate_inputs(domain_info, path, name_file="info"):
    # ATTENTION! Lots of hardcoding
    temperature_info = domain_info["Labels"]["Temperature [C]"]
    shortened_input = {
        "Temperature prediction (1st HP) [C]": temperature_info.copy(),
        "Temperature prediction (other HPs) [C]": temperature_info.copy(),
    }
    shortened_input["Temperature prediction (1st HP) [C]"]["index"] = 0
    shortened_input["Temperature prediction (other HPs) [C]"]["index"] = 1
    shortened_infos = {
        "Inputs": shortened_input,
        "Labels": domain_info["Labels"],
        "CellsNumber": domain_info["CellsNumber"],
        "CellsNumberPrior": domain_info["CellsNumberPrior"],
        "CellsSize": domain_info["CellsSize"],
    }
    save_yaml(shortened_infos, path, name_file)

def save_config_of_merged_inputs(separate_info, path, name_file="info"):
    # ATTENTION! Lots of hardcoding
    temperature_info = separate_info["Labels"]["Temperature [C]"]
    shortened_input = {
        "Temperature prediction (merged) [C]": temperature_info.copy(),
    }
    shortened_input["Temperature prediction (merged) [C]"]["index"] = 0
    shortened_infos = {
        "Inputs": shortened_input,
        "Labels": separate_info["Labels"],
        "CellsNumber": separate_info["CellsNumber"],
        "CellsNumberPrior": separate_info["CellsNumberPrior"],
        "CellsSize": separate_info["CellsSize"],
    }
    save_yaml(shortened_infos, path, name_file)

def check_all_datasets_prepared(paths: List):
    # check if all datasets required are prepared ( domain and 2hp-nn dataset )
    for path in paths:
        if not os.path.exists(path):
            # error
            raise ValueError(f"Dataset {path} not prepared")
        else:
            print(f"Dataset {path} prepared")
