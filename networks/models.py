import logging
import sys
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from networks.dummy_network import DummyCNN, DummyNet
from networks.turbnet import TurbNetG
from networks.unet import UNet


def create_model(model_choice: str, in_channels: int):
    """ takes model_choice-string and returns model """
    if model_choice == "unet":
        model = UNet(in_channels=in_channels, out_channels=2, depth=3, kernel_size=5).float()
    elif model_choice == "fc":
        model = DummyNet(in_channels=in_channels, out_channels=1, size=(128,40)).float()
    elif model_choice == "cnn":
        model = DummyCNN(in_channels=in_channels, out_channels=1).float()
    elif model_choice == "turbnet":
        model = TurbNetG(channelExponent=4, in_channels=in_channels, out_channels=1).float()
    else:
        logging.error("model choice not recognized")
        sys.exit()
    return model

def load_model(model_params:Dict, path:str, file_name:str="model", device:str="cuda:0") -> nn.Module:
    model = create_model(**model_params)
    model.load_state_dict(torch.load(f"{path}/{file_name}.pt", map_location=torch.device(device)))
    return model

def compare_models(model_1, model_2):
    # source: https://discuss.pytorch.org/t/check-if-models-have-same-weights/4351/3
    try:
        # can handle both: model2 being only a state_dict or a full model
        model_2 = model_2.state_dict()
    except:
        pass    
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                logging.warning('Mismatch found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        logging.warning('Models match perfectly! :)')