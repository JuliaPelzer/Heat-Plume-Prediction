from copy import deepcopy
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm
import yaml

import extend_plumes.extend_plumes as ep
from preprocessing.domain_classes.domain import Domain
from preprocessing.domain_classes.heat_pump import HeatPumpBox
from preprocessing.prepare import (prepare_data_and_paths,
                                   prepare_paths_and_settings, prepare_data)
from processing.networks.unet import UNet
from processing.networks.unetVariants import UNetHalfPad2
from utils.utils_data import SettingsTraining, get_run_ids


def preprocessing_allin1(settings: SettingsTraining):
    allin1_paths, settings, _ = prepare_paths_and_settings(settings)
    print("Paths for allin1 prepared")

    if not allin1_paths.dataset_1st_prep_path.exists() or settings.case == "test":
        if "n" in settings.inputs:
            # preprocessing with neural network: 1hpnn(+extend_plumes)
            args_1hpnn = {
                "model": Path("/home/pelzerja/pelzerja/test_nn/1HP_NN/runs/1hpnn/dataset_small_1000dp_varyK_v2 inputs_gksi case_train box256 skip32"), 
                #Path("/home/pelzerja/pelzerja/test_nn/1HP_NN/runs/1hpnn/vary_k/dataset_medium_100dp_vary_perm inputs_gksi case_train box256 skip256 UNet"),
                "inputs": "gksi",
            }

            # prepare allin1 data with 1hp boxes
            allin1_settings = deepcopy(settings)
            settings.case = "test"
            settings.inputs = args_1hpnn["inputs"]
            settings.model = args_1hpnn["model"]
            settings.dataset_prep = ""
            settings.destination = ""
            settings = prepare_data_and_paths(settings)

            args_extend2 = {
                "model": Path("/home/pelzerja/pelzerja/test_nn/1HP_NN/runs/extend_plumes/vary_k/dataset_medium_100dp_vary_perm inputs_gk case_train box128 skip2"), #test_overlap_input_T"),
                "inputs": "gk",
                "box_size": 128,
                "start_prior_box": 64, # box_size // 2
                "skip_per_dir": 4,
                "skip_in_field": 32, #< actual_len
                "overlap": 46, # manually chosen for UNetHalfPad2
            }

            run_id:int = get_run_ids(settings.dataset_prep / "Inputs")[0]
            prediction_destination = f"1hpnn_RUN_{run_id}"

            if (settings.dataset_prep / "Inputs" / f"domain_prediction_{prediction_destination}.pt").exists():
                print(f"Loading domain prediction from 1hpnn from {settings.dataset_prep / 'Inputs' / f'domain_prediction_{prediction_destination}.pt'}")
                # load domain prediction from 1hpnn
                additional_input = torch.load(settings.dataset_prep / "Inputs" / f"domain_prediction_{prediction_destination}.pt")
                additional_input = additional_input.detach()

            else:
                model_1hp = UNet(len(args_1hpnn["inputs"]))
                model_1hp.load(args_1hpnn["model"], map_location=torch.device(settings.device)) # TODO tut das auch für CPU? unnötig? torch.device weg?
                model_1hp.to(settings.device)
                model_1hp.eval()

                model_ep = UNetHalfPad2(len(args_extend2["inputs"])+1)
                model_ep.load(args_extend2["model"], map_location=torch.device(settings.device)) # TODO tut das auch für CPU? unnötig? torch.device weg?
                model_ep.to(settings.device)
                model_ep.eval()

                print("Preparing domain for allin1")
                domain = Domain(settings.dataset_prep, stitching_method="max", file_name=f"RUN_{run_id}.pt", problem=settings.problem)
                # assert domain.inputs[1].max() <= 1, "inputs[1] (perm) must be smaller than 1"
                # assert domain.inputs[1].min() >= 0, "inputs[1] (perm) must be greater than 0"

                max_temperature_threshold = domain.norm(10.7, property = "Temperature [C]")

                print("Extracting hp-boxes and applying 1HP-NN")
                single_hps = domain.extract_hp_boxes(settings.device)
                hp: HeatPumpBox
                for hp in tqdm(single_hps, desc="Applying 1HPNN+extend-plumes and adding arbitrary long boxes to domain"):
                    use_1hp_groundtruth = False
                    if use_1hp_groundtruth:
                        hp.primary_temp_field = hp.label.squeeze(0)
                    else:
                        hp.primary_temp_field = hp.apply_nn(model_1hp, device=settings.device)
                    # assert hp.primary_temp_field.max() <= 1, "primary_temp_field must be smaller than 1 - comes from input[1] (perm) being out of range??"

                    args_extend2["start_prior_box"] = max(args_extend2["start_prior_box"], args_extend2["skip_per_dir"])
                    args_extend2["start_curr_box"] = ep.set_start_curr_box(args_extend2["start_prior_box"], args_extend2)

                    # extend plumes, adapted from ep.infer_nopad inner part
                    counter = 0
                    while hp.primary_temp_field[-1].max() > max_temperature_threshold: # check if extension needed
                        # hp.extend_plumes()
                        # try:
                        input_all, corner_ll, _ = domain.extract_ep_box(hp, args_extend2, device=settings.device)

                        # apply extend plumes model
                        output = model_ep(input_all)
                        actual_len, _ = ep.calc_actual_len_and_gap(output, args_extend2)
                        args_extend2 = correct_skip_in_field(args_extend2, actual_len)

                        insert_at = args_extend2["start_prior_box"] + args_extend2["box_size"] + corner_ll[0]

                        # combine output with hp.primary_temp_field
                        hp.insert_extended_plume(output[0, 0], insert_at, actual_len, device=settings.device)
                        # TODO MODELLVORHERSAGE IST SHITTY, klar, weil perm-freq. passt ja nicht - trotzdem checken

                            # increase counter
                            args_extend2["start_prior_box"] += args_extend2["skip_in_field"]
                            start_curr_box = ep.set_start_curr_box(args_extend2["start_prior_box"], args_extend2)
                        except:
                            print("Error in extension, e.g. box extending outside of domain")
                            break
                    # else:
                    #     print("All good")

                    domain.add_hp(hp, hp.primary_temp_field) # includes does reverse_norm acc. to domain

                # save domain, # TODO where / how to save
                if len(domain.prediction.shape) == 2:
                    domain.prediction = domain.prediction.unsqueeze(0)
                print(f"Saving domain of size {domain.prediction.shape} to {settings.dataset_prep / 'Inputs' / f'RUN_{run_id}_prediction_1hpnn.pt'}")
                domain.save(folder=settings.dataset_prep / "Inputs", name=prediction_destination)
                additional_input = domain.prediction.detach()
        else:
            additional_input = None

        # then prepare domain for allin1
        settings = prepare_data(allin1_settings, allin1_paths, inputs_2hp=None, additional_input=additional_input)

    return settings

def correct_skip_in_field(args_extend2, actual_len):
    if actual_len < args_extend2["skip_in_field"]: 
        skip_in_field = actual_len
        print(f"Changed skip_in_field to {skip_in_field} because actual_len is smaller ({actual_len}).")
    
    return args_extend2