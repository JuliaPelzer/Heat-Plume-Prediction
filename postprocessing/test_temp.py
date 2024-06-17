import logging
import os
import pathlib
from typing import List
import time
import yaml
import numpy as np

from torch import unsqueeze
from tqdm.auto import tqdm

from networks.unet import UNet
from postprocessing.visualization import (DataToVisualize,
                                          plot_datafields)
from data_stuff.utils import SettingsTraining
from preprocessing.domain_classes.domain import Domain
from preprocessing.domain_classes.heat_pump import HeatPumpBox
from data_stuff.transforms import NormalizeTransform

def test_temp(model: UNet, settings: SettingsTraining):
    time_start_prep_2hp = time.perf_counter()

    #prepare data and normalization
    list_runs = os.listdir(settings.dataset_prep / "Inputs")
    settings_pic = {"format": "png",
                "dpi": 600,}  
    info_file_path = settings.model / "info.yaml"
    with open(info_file_path, "r") as file:
         info = yaml.safe_load(file)
    norm = NormalizeTransform(info)

    model.eval()  

    for run_file in tqdm(list_runs, desc="2HP prepare", total=len(list_runs)):
        # for each run, configure save destination
        run_id = f'{run_file.split(".")[0]}'
        pathlib.Path(settings.destination / run_id).mkdir(parents=True, exist_ok=True)

        # load domain
        domain = Domain(settings.dataset_prep, stitching_method="update", file_name=run_file)
        if domain.skip_datapoint:
            logging.warning(f"Skipping {run_id}")
            continue

        # generate 1hp-boxes and extract information like perm and ids etc.
        single_hps = domain.extract_hp_boxes("cpu")

        for current in range(len(single_hps)):
            # visualization purpose
            dict_to_plot = {}
            name_pic = settings.destination / run_id / str(current)

            #apply NN
            time_start_run_1hp = time.perf_counter()
            x = unsqueeze(single_hps[current].inputs.to("cpu"), 0)
            single_hps[current].primary_temp_field = model(x).squeeze().detach()

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
            dict_to_plot[f"t_out_{current}"] = DataToVisualize(y, f"Prediction hp{current}: Temperature in [°C]", extent_highs, {"vmax": temp_max, "vmin": temp_min})

            # plot 
            plot_datafields(dict_to_plot, name_pic, settings_pic)

            #update domain
            domain.add_hp(single_hps[current])
            single_hps = domain.extract_hp_boxes("cpu")
            
        #plot final domain
        domain.plot("t",settings.destination / run_id ,"domain_step_final")

    print(f"visualization saved in: {settings.destination}")
    time_end = time.perf_counter()



