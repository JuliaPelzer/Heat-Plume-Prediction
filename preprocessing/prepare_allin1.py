from copy import deepcopy
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm
import yaml

from preprocessing.domain_classes.domain import Domain
from preprocessing.domain_classes.heat_pump import HeatPumpBox
from preprocessing.prepare import (prepare_data_and_paths,
                                   prepare_paths_and_settings, prepare_data)
from processing.networks.unet import UNet
from processing.networks.unetVariants import UNetHalfPad2
from utils.utils_data import SettingsTraining


def preprocessing_allin1(settings: SettingsTraining):
    allin1_paths, settings, _ = prepare_paths_and_settings(settings)
    print("Paths for allin1 prepared")

    if not allin1_paths.dataset_1st_prep_path.exists() or settings.case == "test":
        if "n" in settings.inputs:
            # preprocessing with neural network: 1hpnn(+extend_plumes)
            run_id:int = 0
            prediction_destination = f"1hpnn_RUN_{run_id}"
            # timestep = "   3 Time  5.00000E+00 y" # '   4 Time  2.75000E+01 y'
            args_1hpnn = {
                "model": Path("/home/pelzerja/Development/DaRUS_DATA_MODELS_extend_plumes/MODELS/extend1/dataset_medium_100dp_vary_perm inputs_gksi case_train box256 skip256 UNet"),
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

            if (settings.dataset_prep / "Inputs" / f"domain_prediction_{prediction_destination}.pt").exists():
                print(f"Loading domain prediction from 1hpnn from {settings.dataset_prep / 'Inputs' / f'domain_prediction_{prediction_destination}.pt'}")
                # load domain prediciton from 1hpnn
                additional_input = torch.load(settings.dataset_prep / "Inputs" / f"domain_prediction_{prediction_destination}.pt")
                additional_input = additional_input.detach()
            else:

                model_1hp = UNet(len(args_1hpnn["inputs"]))
                model_1hp.load(args_1hpnn["model"], map_location=torch.device(settings.device))
                model_1hp.eval()

                # define domain (allin1)
                print("Preparing domain for allin1")
                domain = Domain(settings.dataset_prep, stitching_method="max", file_name=f"RUN_{run_id}.pt")
                # extract hp-boxes and apply 1hp-NN
                print("Extracting hp-boxes and applying 1hp-NN")
                single_hps = domain.extract_hp_boxes(settings.device)
                hp: HeatPumpBox
                for hp in tqdm(single_hps, desc="Applying NN and adding boxes to domain"):
                    # start timer
                    hp.primary_temp_field = hp.apply_nn(model_1hp)
                    # TODO apply extend-plumes
                    domain.add_hp(hp, hp.primary_temp_field) # includes does reverse_norm acc. to domain

                # save domain,
                # TODO where / how to save
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