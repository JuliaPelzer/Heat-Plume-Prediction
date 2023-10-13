import argparse
import logging
import os
import pathlib
import sys
import time
import yaml

from torch import tensor, stack, load
from tqdm.auto import tqdm

from data_stuff.utils import load_yaml
from networks.unet import UNet
from preprocessing.prepare_1ststage import prepare_dataset

sys.path.append("/home/pelzerja/pelzerja/test_nn/1HP_NN")  # relevant for remote
sys.path.append("/home/pelzerja/Development/1HP_NN")  # relevant for local
sys.path.append("/home/pelzerja/pelzerja/test_nn/2HPs_demonstrator")  # relevant for remote
sys.path.append("/home/pelzerja/Development/2HPs_demonstrator/2HPs_demonstrator")  # relevant for remote
from domain import Domain
from heat_pump import HeatPump
from utils_2hp import save_config_of_separate_inputs
from utils.prepare_paths import Paths2HP


def prepare_dataset_for_2nd_stage(paths: Paths2HP, dataset_name: str, inputs_1hp: str, device: str = "cuda:0"):
    """
    assumptions:
    - 1hp-boxes are generated already
    - 1hpnn is trained
    - cell sizes of 1hp-boxes and domain are the same
    - boundaries of boxes around at least one hp is within domain
    - device: attention, all stored need to be produced on cpu for later pin_memory=True and all other can be gpu
    """
    
    timestamp_begin = time.ctime()
    time_begin = time.perf_counter()

# prepare domain dataset if not yet done
    ## load model from 1st stage
    time_start_prep_domain = time.perf_counter()
    model_1HP = UNet(in_channels=len(inputs_1hp)).float()
    model_1HP.load_state_dict(load(f"{paths.model_1hp_path}/model.pt", map_location=device))
    # model_1HP.to(device)
    
    ## prepare 2hp dataset for 1st stage
    if not os.path.exists(paths.dataset_1st_prep_path):        
        # norm with data from dataset that NN was trained with!!
        with open(os.path.join(os.getcwd(), paths.dataset_model_trained_with_prep_path, "info.yaml"), "r") as file:
            info = yaml.safe_load(file)
        prepare_dataset(paths, dataset_name, inputs_1hp, info=info, power2trafo=False)
    print(f"Domain {paths.dataset_1st_prep_path} prepared")

# prepare dataset for 2nd stage
    time_start_prep_2hp = time.perf_counter()
    avg_time_inference_1hp = 0
    list_runs = os.listdir(os.path.join(paths.dataset_1st_prep_path, "Inputs"))
    for run_file in tqdm(list_runs, desc="2HP prepare", total=len(list_runs)):
        run_id = f'{run_file.split(".")[0]}_'
        domain = Domain(paths.dataset_1st_prep_path, stitching_method="max", file_name=run_file)
        ## generate 1hp-boxes and extract information like perm and ids etc.
        if domain.skip_datapoint:
            logging.warning(f"Skipping {run_id}")
            continue

        single_hps = domain.extract_hp_boxes()
        # apply learned NN to predict the heat plumes
        hp: HeatPump
        for hp in single_hps:
            time_start_run_1hp = time.perf_counter()
            hp.primary_temp_field = hp.apply_nn(model_1HP)
            avg_time_inference_1hp += time.perf_counter() - time_start_run_1hp
            hp.primary_temp_field = domain.reverse_norm(hp.primary_temp_field, property="Temperature [C]")
        avg_time_inference_1hp /= len(single_hps)

        for hp in single_hps:
            hp.get_other_temp_field(single_hps)

        for hp in single_hps:
            hp.primary_temp_field = domain.norm(hp.primary_temp_field, property="Temperature [C]")
            hp.other_temp_field = domain.norm(hp.other_temp_field, property="Temperature [C]")
            inputs = stack([hp.primary_temp_field, hp.other_temp_field])
            hp.save(run_id=run_id, dir=paths.datasets_boxes_prep_path, inputs_all=inputs,)

    time_end = time.perf_counter()
    avg_inference_times = avg_time_inference_1hp / len(list_runs)

    # save infos of info file about separated (only 2!) inputs
    save_config_of_separate_inputs(domain.info, path=paths.datasets_boxes_prep_path)

    # save measurements
    with open(os.path.join(os.getcwd(), "runs", paths.datasets_boxes_prep_path, f"measurements.yaml"), "w") as f:
        f.write(f"timestamp of beginning: {timestamp_begin}\n")
        f.write(f"timestamp of end: {time.ctime()}\n")
        f.write(f"model 1HP: {paths.model_1hp_path}\n")
        f.write(f"input params: {inputs_1hp}\n")
        f.write(f"separate inputs: {True}\n")
        f.write(f"location of prepared domain dataset: {paths.dataset_1st_prep_path}\n")
        f.write(f"name of dataset prepared with: {paths.dataset_model_trained_with_prep_path}\n")
        f.write(f"name of dataset domain: {dataset_name}\n")
        f.write(f"name_destination_folder: {paths.datasets_boxes_prep_path}\n")
        f.write(f"avg inference times for 1HP-NN in seconds: {avg_inference_times}\n")
        f.write(f"device: {device}\n")
        f.write(f"duration of preparing domain in seconds: {(time_start_prep_2hp-time_start_prep_domain)}\n")
        f.write(f"duration of preparing 2HP in seconds: {(time_end-time_start_prep_2hp)}\n")
        f.write(f"duration of preparing 2HP /run in seconds: {(time_end-time_start_prep_2hp)/len(list_runs)}\n")
        f.write(f"duration of whole process in seconds: {(time_end-time_begin)}\n")
