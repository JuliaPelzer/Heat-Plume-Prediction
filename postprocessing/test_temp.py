import argparse
import logging
import os
import pathlib
from typing import List
import time
import yaml
import numpy as np

from torch.utils.data import DataLoader
from torch import unsqueeze,save,load
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
from data_stuff.transforms import NormalizeTransform

def test_temp(model: UNet, settings: SettingsTraining):
    time_start_prep_2hp = time.perf_counter()

    avg_time_inference_1hp = 0
    list_runs = os.listdir(settings.dataset_prep / "Inputs")
    model.eval()
    settings_pic = {"format": "png",
                "dpi": 600,}  
    info_file_path = settings.model / "info.yaml"
    with open(info_file_path, "r") as file:
         info = yaml.safe_load(file)
    norm = NormalizeTransform(info)     
    for run_file in tqdm(list_runs, desc="2HP prepare", total=len(list_runs)):
        # for each run, configure domain
        run_id = f'{run_file.split(".")[0]}'
        pathlib.Path(settings.destination / run_id).mkdir(parents=True, exist_ok=True)
        domain = Domain(settings.dataset_prep, stitching_method="update", file_name=run_file)

        ## generate 1hp-boxes and extract information like perm and ids etc.
        if domain.skip_datapoint:
            logging.warning(f"Skipping {run_id}")
            continue

        single_hps = domain.extract_hp_boxes("cpu")

        for current in range(len(single_hps)):

            dict_to_plot = {}
            name_pic = settings.destination / run_id / str(current)
            #apply NN
            time_start_run_1hp = time.perf_counter()
            x = unsqueeze(single_hps[current].inputs.to("cpu"), 0)
            single_hps[current].primary_temp_field = model(x).squeeze().detach()
            avg_time_inference_1hp += time.perf_counter() - time_start_run_1hp

            #plot domain
            domain.plot("t",settings.destination / run_id ,f"domain_step_{current}")

            #plot inputs
            extent_highs = (np.array(info["CellsSize"][:2]) * single_hps[current].inputs.shape[-2:])
            inputs = single_hps[current].inputs.clone().detach()
            inputs = norm.reverse(inputs.cpu(), "Inputs")
            inputKeys = info["Inputs"].keys()
            for input in inputKeys:
                index = info["Inputs"][input]["index"]
                dict_to_plot[f"{input}_{current}"] = DataToVisualize(inputs[index], f"step {current} : {input}", extent_highs)

            #plot output
            output = single_hps[current].primary_temp_field.clone().detach()
            y = domain.reverse_norm(output.cpu())
            temp_max = y.max()
            temp_min = y.min()
            print(f"mean of output: {y.mean()}")
            print(f"min Temp Output: {temp_min}")
            print(f"max Temp Output: {temp_max}")
            dict_to_plot[f"t_out_{current}"] = DataToVisualize(y, f"Prediction hp{current}: Temperature in [°C]", extent_highs, {"vmax": temp_max, "vmin": temp_min})

            #update domain
            #hp.get_other_temp_field(single_hps)
            domain.add_hp(single_hps[current])
            
            single_hps = domain.extract_hp_boxes("cpu")
            plot_datafields(dict_to_plot, name_pic, settings_pic)
        #plot final domain
        domain.plot("t",settings.destination / run_id ,"domain_step_final")

        name_pic = settings.destination / run_id / "final"
        plot_datafields(dict_to_plot, name_pic, settings_pic)
        avg_time_inference_1hp /= len(single_hps)
    print(f"visualization saved in: {settings.destination}")
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



