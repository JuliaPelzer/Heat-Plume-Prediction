import argparse
import logging
import os
import pathlib
import sys
import time
import torch

import numpy as np
from tqdm.auto import tqdm

from data_stuff.utils import SettingsPrepare, load_yaml, save_yaml
from networks.unet import UNet
from prepare_1ststage import prepare_dataset

sys.path.append("/home/pelzerja/pelzerja/test_nn/1HP_NN")  # relevant for remote
sys.path.append("/home/pelzerja/Development/1HP_NN")  # relevant for local
sys.path.append("/home/pelzerja/pelzerja/test_nn/2HPs_demonstrator")  # relevant for remote
from domain import Domain
from heat_pump import HeatPump
from utils_2hp import save_config_of_separate_inputs, set_paths


def prepare_inputs_for_2nd_stage(
    dataset_large_name: str,
    preparation_case: str,
    device: str = "cuda:0",
):
    """
    assumptions:
    - 1hp-boxes are generated already
    - 1hpnn is trained
    - cell sizes of 1hp-boxes and domain are the same
    - boundaries of boxes around at least one hp is within domain
    """
    timestamp_begin = time.ctime()
    time_begin = time.perf_counter()

# prepare domain dataset if not yet done
    (datasets_raw_domain_dir, datasets_prepared_domain_dir, dataset_domain_path, datasets_model_trained_with_path, model_1hp_path, _, datasets_prepared_2hp_dir, destination_2hp_prep, inputs_prep) = set_paths(dataset_large_name, preparation_case)

    ## load model from 1st stage
    model_1HP = UNet(in_channels=len(inputs_prep)).float()
    model_1HP.load_state_dict(torch.load(f"{model_1hp_path}/model.pt", map_location=torch.device(device)))
    model_1HP.to(device)

    ## prepare domain dataset 
    time_start_prep_domain = time.perf_counter()
    if not os.path.exists(dataset_domain_path):
        args = SettingsPrepare(
            raw_dir=datasets_raw_domain_dir,
            datasets_dir=datasets_prepared_domain_dir,
            dataset_name=dataset_large_name,
            inputs_prep=inputs_prep,)
        prepare_dataset(args, datasets_prepared_2hp_dir, power2trafo=False, info=load_yaml(datasets_model_trained_with_path, "info"),
        )  # norm with data from dataset that NN was trained with!!
    print(f"Domain {dataset_domain_path} prepared")
    
# prepare dataset for 2nd stage
    time_start_prep_2hp = time.perf_counter()
    avg_time_inference_1hp = 0
    list_runs = os.listdir(os.path.join(dataset_domain_path, "Inputs"))
    for run_file in tqdm(list_runs, desc="2HP prepare", total=len(list_runs)):
        run_id = f'{run_file.split(".")[0]}_'
        domain = Domain(dataset_domain_path, stitching_method="max", file_name=run_file)
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
            inputs = np.array([hp.primary_temp_field, hp.other_temp_field]),
            hp.save(run_id=run_id, dir=destination_2hp_prep, inputs_all=inputs,)

    time_end = time.perf_counter()
    avg_inference_times = avg_time_inference_1hp / len(list_runs)

    # save infos of info file about separated (only 2!) inputs
    save_config_of_separate_inputs(
        domain.info, path=destination_2hp_prep, name_file="info"
    )
    # save command line arguments
    cla = {
        "dataset_large_name": dataset_large_name,
        "preparation_case": preparation_case,
    }
    save_yaml(cla, path=destination_2hp_prep, name_file="command_line_args")

    # save measurements
    with open(os.path.join(os.getcwd(), "runs", destination_2hp_prep, f"measurements.yaml"), "w") as f:
        f.write(f"timestamp of beginning: {timestamp_begin}\n")
        f.write(f"timestamp of end: {time.ctime()}\n")
        f.write(f"model 1HP: {model_1hp_path}\n")
        f.write(f"input params: {inputs_prep}\n")
        f.write(f"separate inputs: {True}\n")
        f.write(f"dataset prepared location: {datasets_prepared_domain_dir}\n")
        f.write(f"dataset name: {datasets_model_trained_with_path}\n")
        f.write(f"dataset large name: {dataset_large_name}\n")
        f.write(f"name_destination_folder: {destination_2hp_prep}\n")
        f.write(f"avg inference times for 1HP-NN in seconds: {avg_inference_times}\n")
        f.write(f"device: {device}\n")
        f.write(f"duration of preparing domain in seconds: {(time_start_prep_2hp-time_start_prep_domain)}\n")
        f.write(f"duration of preparing 2HP in seconds: {(time_end-time_start_prep_2hp)}\n")
        f.write(f"duration of preparing 2HP /run in seconds: {(time_end-time_start_prep_2hp)/len(list_runs)}\n")
        f.write(f"duration of whole process in seconds: {(time_end-time_begin)}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preparation_case", type=str, default="gksi_100dp")
    parser.add_argument("--dataset_large", type=str, default="benchmark_dataset_2d_2hps_iso_perm")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    assert args.preparation_case in ["gksi_100dp", "ogksi_1000dp", "gksi_1000dp", "pksi_100dp", "pksi_1000dp", "ogksi_1000dp_finetune"], "preparation_case must be one of ['gksi_100dp', 'gksi_1000dp', 'pksi_100dp', 'pksi_1000dp', 'ogksi_1000dp_finetune']"

    prepare_inputs_for_2nd_stage(
        dataset_large_name=args.dataset_large,
        preparation_case=args.preparation_case,
        device=args.device,
    )