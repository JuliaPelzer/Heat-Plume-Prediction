import argparse
import logging
import os
import pathlib
from typing import List
import time
import yaml
import numpy as np

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from networks.unet import UNet
from postprocessing.visualization import (infer_all_and_summed_pic,
                                          plot_avg_error_cellwise,
                                          visualizations,
                                          DataToVisualize,
                                          plot_datafields)
from data_stuff.utils import SettingsTraining
from preprocessing.domain_classes.domain import Domain
from preprocessing.domain_classes.heat_pump import HeatPumpBox
from preprocessing.domain_classes.stitching import Stitching
from preprocessing.domain_classes.utils_2hp import (
    save_config_of_merged_inputs, save_config_of_separate_inputs, save_yaml)


def test_temp(model: UNet, dataloader: DataLoader, settings: SettingsTraining):

    #load data
    #if not os.path.exists(paths.dataset_1st_prep_path):        
        # norm with data from dataset that NN was trained with!!
    #    with open(paths.dataset_model_trained_with_prep_path / "info.yaml", "r") as file:
    #        info = yaml.safe_load(file)
    #    prepare_dataset(paths, settings.inputs, info=info, power2trafo=False) # for using unet on whole domain required: power2trafo=True
    #print(f"Domain prepared ({paths.dataset_1st_prep_path})")
    
    # prepare dataset for 2nd stage
    time_start_prep_2hp = time.perf_counter()
    avg_time_inference_1hp = 0
    list_runs = os.listdir(settings.dataset_prep / "Inputs")
    settings_pic = {"format": pic_format,
                "dpi": 600,}

    info_file_path = settings.model / "info.yaml"
    with open(info_file_path, "r") as file:
         info = yaml.safe_load(file)

    for run_file in tqdm(list_runs, desc="2HP prepare", total=len(list_runs)):
        # for each run, configure domain
        run_id = f'{run_file.split(".")[0]}_'
        domain = Domain(settings.dataset_prep, stitching_method="update", file_name=run_file)
        prediction_field = domain

        ## generate 1hp-boxes and extract information like perm and ids etc.
        if domain.skip_datapoint:
            logging.warning(f"Skipping {run_id}")
            continue

        single_hps = domain.extract_hp_boxes("cpu")
        for current in len(single_hps):
            #apply NN
            time_start_run_1hp = time.perf_counter()
            single_hps[current].primary_temp_field = single_hps[current].apply_nn(model)
            avg_time_inference_1hp += time.perf_counter() - time_start_run_1hp

            temp_max = single_hps[current].primary_temp_field.max()
            temp_min = single_hps[current].primary_temp_field.min()
            extent_highs = (np.array(info["CellsSize"][:2]) * single_hps[current].inputs.shape[-2:])
            dict_to_plot = {
                "t_out": DataToVisualize(single_hps[current].primary_temp_field, f"Prediction hp{current}: Temperature in [°C]", extent_highs, {"vmax": temp_max, "vmin": temp_min}),
            }
            name_pic = f"{settings.destination}_{run_id}_hp{current}"
            plot_datafields(dict_to_plot, name_pic, settings_pic)
            #update domain
            #hp.get_other_temp_field(single_hps)
            domain.add_hp(single_hps[current])
            single_hps = domain.extract_hp_boxes("cpu")
        avg_time_inference_1hp /= len(single_hps)
    print(f"visualization saved in: {name_pic}")
    time_end = time.perf_counter()
    avg_inference_times = avg_time_inference_1hp / len(list_runs)

    # save infos of info file about separated (only 2!) inputs
    #save_config_of_separate_inputs(domain.info, path=paths.datasets_boxes_prep_path)



def prepare_hp_boxes(settings: SettingsTraining, model_1HP:UNet, single_hps:List[HeatPumpBox], domain:Domain, run_id:int, avg_time_inference_1hp:float=0, save_bool:bool=True):
    hp: HeatPumpBox
    for hp in single_hps:
        #apply NN
        time_start_run_1hp = time.perf_counter()
        hp.primary_temp_field = hp.apply_nn(model_1HP)
        avg_time_inference_1hp += time.perf_counter() - time_start_run_1hp

        #update domain
        #hp.get_other_temp_field(single_hps)
        domain.add_hp(hp)
    avg_time_inference_1hp /= len(single_hps)
    return single_hps, avg_time_inference_1hp



