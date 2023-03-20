import sys
import logging
import numpy as np
from typing import Dict
import torch.nn as nn
import torch
from networks.unet import UNet
from networks.turbnet import TurbNetG
from networks.dummy_network import DummyNet, DummyCNN

def create_model(model_choice: str, in_channels: int):
    """ takes model_choice-string and returns model """
    if model_choice == "unet":
        model = UNet(in_channels=in_channels, out_channels=1, depth=3, kernel_size=5).float()
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

def load_model(model_params:Dict, path:str, file_name:str="model") -> nn.Module:
    model = create_model(**model_params)
    model.load_state_dict(torch.load(f"{path}/{file_name}.pt"))
    return model