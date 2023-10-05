import argparse
import os
import pathlib
import sys
import time
import torch
import yaml

sys.path.append("/home/pelzerja/pelzerja/test_nn/2HPs_demonstrator")  # relevant for remote
from stitching import Stitching
from utils_2hp import save_config_of_merged_inputs, save_yaml

def merge_inputs_for_2HPNN(path_separate_inputs:pathlib.Path, path_merged_inputs:pathlib.Path, stitching_method:str="max"):
    begin = time.perf_counter()
    assert stitching_method == "max", "Other than max stitching required reasonable background temp and therefor potentially norming."
    stitching = Stitching(stitching_method, background_temperature=0)
    
    (path_merged_inputs/"Inputs").mkdir(exist_ok=True)

    begin_prep = time.perf_counter()
    # get separate inputs if exist
    for file in (path_separate_inputs/"Inputs").iterdir():
        input = torch.load(file)
        # merge inputs via stitching
        input = stitching(input[0], input[1])
        # save merged inputs
        input = torch.unsqueeze(torch.Tensor(input), 0)
        torch.save(input, path_merged_inputs/"Inputs"/file.name)
    end_prep = time.perf_counter()

    # save config of merged inputs
    info_separate = yaml.load(open(path_separate_inputs/"info.yaml", "r"), Loader=yaml.FullLoader)
    save_config_of_merged_inputs(info_separate, path_merged_inputs)

    # save command line arguments
    cla = {
        "dataset_separate": path_separate_inputs.name,
        "command": "prepare_2HP_merged_inputs.py"
    }
    save_yaml(cla, path=path_merged_inputs, name_file="command_line_args")
    end = time.perf_counter()

    # save times in measurements.yaml (also copy relevant ones from separate)
    measurements_prep_separate = yaml.load(open(path_separate_inputs/"measurements.yaml", "r"), Loader=yaml.FullLoader)
    num_dp = len(list((path_separate_inputs/"Inputs").iterdir()))
    duration_prep = end_prep - begin_prep
    duration_prep_avg = duration_prep / num_dp
    measurements = {
        "duration of preparation in seconds": duration_prep,
        "duration of preparing 2HP /run in seconds": duration_prep_avg,
        "duration total in seconds": end - begin,
        "number of datapoints": num_dp,
        "separate-preparation": {"duration of preparing domain in seconds": measurements_prep_separate["duration of preparing domain in seconds"],
                                    "duration of preparing 2HP /run in seconds": measurements_prep_separate["duration of preparing 2HP /run in seconds"],
                                    "duration of preparing 2HP in seconds": measurements_prep_separate["duration of preparing 2HP in seconds"],
                                    "duration of whole process in seconds": measurements_prep_separate["duration of whole process in seconds"]},
    }
    save_yaml(measurements, path=path_merged_inputs, name_file="measurements")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dataset_2hps_1fixed_10dp_2hp_gksi_1000dp")
    args = parser.parse_args()

    #get dir of prepare_2HP_separate_inputs
    paths = yaml.load(open("paths.yaml", "r"), Loader=yaml.FullLoader)
    dir_separate_inputs = paths["datasets_prepared_2hp_dir"]

    path_separate_inputs = pathlib.Path(dir_separate_inputs) / args.dataset
    path_merged_inputs = pathlib.Path(dir_separate_inputs) / (args.dataset+"_merged")
    path_merged_inputs.mkdir(exist_ok=True)
    # copy "Labels" folder from separate to merged
    os.system(f"cp -r {path_separate_inputs/'Labels'} {path_merged_inputs}")

    if os.path.exists(path_separate_inputs):
        merge_inputs_for_2HPNN(path_separate_inputs, path_merged_inputs, stitching_method="max")
    else:
        print(f"Could not find prepared dataset with separate inputs at {path_separate_inputs}.")