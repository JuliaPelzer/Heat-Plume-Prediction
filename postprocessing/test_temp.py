import logging
import os
import pathlib
from typing import List
import time
import yaml
import numpy as np

from torch import unsqueeze, stack
import torch
from tqdm.auto import tqdm

from networks.unet import UNet
from postprocessing.visualization import (DataToVisualize,
                                          plot_datafields)
from data_stuff.utils import SettingsTraining
from preprocessing.domain_classes.domain import Domain
from preprocessing.domain_classes.heat_pump import HeatPumpBox
from data_stuff.transforms import NormalizeTransform

batch = False
save = True

def test_temp(model: UNet, settings: SettingsTraining):
    time_start_prep_2hp = time.perf_counter()

    #prepare data and normalization
    list_runs = os.listdir(settings.dataset_prep / "Inputs")
    settings_pic = {"format": "png",
                "dpi": 600,}  
    info_file_path = settings.model / "info.yaml"
    with open(info_file_path, "r") as file:
         info = yaml.safe_load(file)

    model.eval()  
    index = 0
    for run_file in tqdm(list_runs, desc="2HP prepare", total=len(list_runs)):
        if index > 50:
            break
        index += 1
        # for each run, configure save destination
        run_id = f'{run_file.split(".")[0]}'
        pathlib.Path(settings.destination / run_id).mkdir(parents=True, exist_ok=True)

        # load domain
        if batch:
            domain = Domain(settings.dataset_prep, stitching_method="max", file_name=run_file)
        else:
            domain = Domain(settings.dataset_prep, stitching_method="update", file_name=run_file)
        if domain.skip_datapoint:
            logging.warning(f"Skipping {run_id}")
            continue
        
        dict_to_plot = {}
        norm = NormalizeTransform(info)
        extent_highs = (np.array(info["CellsSize"][:2]) * domain.inputs.shape[-2:])
        inputs = domain.inputs.clone().detach()
        inputs = norm.reverse(inputs.cpu(), "Inputs")
        inputKeys = info["Inputs"].keys()
        for input in inputKeys:
            index = info["Inputs"][input]["index"]
            if input == "Temperature prediction (other HPs) [C]":
                dict_to_plot[f"domain_{input}_{run_id}"] = DataToVisualize(inputs[index], f"Temperature in [C]", extent_highs)
            elif input == "SDF":
                continue
            else:
                dict_to_plot[f"domain_{input}_{run_id}"] = DataToVisualize(inputs[index], f"{input}", extent_highs)
        name_pic = settings.destination / run_id / "domain_inputs"
        plot_datafields(dict_to_plot, name_pic, settings_pic)


        if batch:
            apply_batch(model,settings,domain,run_id,info,settings_pic)
        else:
            apply_iterative(model,settings,domain,run_id,info,settings_pic)
            

    print(f"visualization saved in: {settings.destination}")
    time_end = time.perf_counter()

def apply_batch(model: UNet, settings: SettingsTraining, domain, run_id, info, settings_pic):
    norm = NormalizeTransform(info)
    single_hps = domain.extract_hp_boxes("cpu")
    avg_time_inference_all = 0

    for step in range(len(single_hps)):

        #apply NN
        time_start_run_1hp = time.perf_counter()
        x = []
        for current in range(len(single_hps)):
            input = single_hps[current].inputs.clone().detach().to("cpu")
            x.append(input)
        x = torch.stack(x)
        y = model(x).squeeze().detach()
        avg_time_inference = time.perf_counter() - time_start_run_1hp

        avg_time_inference_all += avg_time_inference



        #plot inputs
        for current in range(len(single_hps)):
            # visualization purpose
            dict_to_plot = {}
            pathlib.Path(settings.destination / run_id / str(step)).mkdir(parents=True, exist_ok=True)
            name_pic = settings.destination / run_id / str(step) / str(current)
            current_output = y[current]
            single_hps[current].primary_temp_field = current_output.clone().detach()
            domain.add_hp(single_hps[current])
            extent_highs = (np.array(info["CellsSize"][:2]) * single_hps[current].inputs.shape[-2:])
            inputs = single_hps[current].inputs.clone().detach()
            inputs = norm.reverse(inputs.cpu(), "Inputs")
            inputKeys = info["Inputs"].keys()
            for input in inputKeys:
                index = info["Inputs"][input]["index"]
                dict_to_plot[f"{input}_{current}"] = DataToVisualize(inputs[index], f"step {current} : {input}", extent_highs)


            temp_max = current_output.max()
            temp_min = current_output.min()
            dict_to_plot[f"t_out_{current}"] = DataToVisualize(current_output, f"Prediction hp{current}: Temperature in [°C]", extent_highs, {"vmax": temp_max, "vmin": temp_min})
            label = single_hps[current].label.clone().detach()
            dict_to_plot[f"label_{current}"] = DataToVisualize(label, f"Label: Temperature in [°C]", extent_highs)
            plot_datafields(dict_to_plot, name_pic, settings_pic)

        #plot domain
        domain.plot("t",settings.destination / run_id ,f"domain_step_{step}")

        single_hps = domain.extract_hp_boxes("cpu")


    #plot final domain
    domain.plot("t",settings.destination / run_id ,"domain_step_final")
    with open(settings.destination / run_id / "measurements.yaml", "w") as f:
        f.write(f"avg inference times for 1HP-NN in seconds: {avg_time_inference_all/len(single_hps)}\n")

def apply_iterative(model: UNet, settings: SettingsTraining, domain, run_id, info, settings_pic):
    norm = NormalizeTransform(info)
    single_hps = domain.extract_hp_boxes("cpu")
    avg_time_inference_all = 0
    #plot inital domain
    domain.plot("t",settings.destination / run_id ,"domain_step_init")
    for current in range(len(single_hps)):
        # visualization purpose
        dict_to_plot = {}
        name_pic = settings.destination / run_id / str(current)
        data_path_temp = settings.destination/ "dataset" /str(current) 

        hp = single_hps[current]
        
        #apply NN
        time_start_run_1hp = time.perf_counter()
        x = unsqueeze(single_hps[current].inputs.to("cpu"), 0)
        single_hps[current].primary_temp_field = model(x).squeeze().detach()
        time_inference = time.perf_counter() - time_start_run_1hp
        avg_time_inference_all += time_inference

        hp.inputs[4] = single_hps[current].primary_temp_field.clone().detach()
        inputs = stack([hp.inputs[0],hp.inputs[1],hp.inputs[2],hp.inputs[3], hp.inputs[4]])
        if save:
            hp.save(run_id=run_id, dir=data_path_temp, inputs_all=inputs,)
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
        label = single_hps[current].label.clone().detach()
        dict_to_plot[f"label_{current}"] = DataToVisualize(label, f"Label: Temperature in [°C]", extent_highs)

        # plot 
        plot_datafields(dict_to_plot, name_pic, settings_pic)

        #update domain
        domain.plot("t",settings.destination / run_id ,f"domain_step_pre_{current}",corner_ll=single_hps[current].corner_ll,corner_ur=single_hps[current].corner_ur)
        domain.add_hp(single_hps[current])
        domain.plot("t",settings.destination / run_id ,f"domain_step_{current}",corner_ll=single_hps[current].corner_ll,corner_ur=single_hps[current].corner_ur)
        single_hps = domain.extract_hp_boxes("cpu")

    domain.plot("t",settings.destination / run_id ,f"domain_step_final")
    

    with open(settings.destination / run_id / "measurements.yaml", "w") as f:
        f.write(f"avg inference times for 1HP-NN in seconds step: {avg_time_inference_all/len(single_hps)}\n")