import sys
import logging
import numpy as np
from typing import Dict
import torch.nn as nn
import torch
from networks.unet import TurbNetG, UNet
from networks.dummy_network import DummyNet, DummyCNN

def create_model(model_choice: str, in_channels: int, datasets_2D: dict=None, reduce_to_2D: bool=True):
    """ takes model_choice-string and returns model """
    if model_choice == "unet":
        model = UNet(in_channels=in_channels, out_channels=1, depth=3).float()
    elif model_choice == "fc":
        size_domain_2D = datasets_2D["train"].dimensions_of_datapoint
        if reduce_to_2D:
            # TODO order here or in dummy_network(size) messed up
            size_domain_2D = size_domain_2D[1:]
        # transform to PowerOfTwo
        size_domain_2D = [2 ** int(np.log2(dimension)) for dimension in size_domain_2D]
        model = DummyNet(in_channels=in_channels, out_channels=1, size=size_domain_2D).float()
    elif model_choice == "cnn":
        model = DummyCNN(in_channels=in_channels, out_channels=1).float()
    elif model_choice == "turbnet":
        model = TurbNetG(channelExponent=4, in_channels=in_channels, out_channels=1).float()
    else:
        logging.error("model choice not recognized")
        sys.exit()
    # print(model)
    return model

def load_model(model_params:Dict, path:str, file_name:str="model") -> nn.Module:
    model = create_model(**model_params)
    model.load_state_dict(torch.load(f"{path}/{file_name}.pt"))
    return model

if __name__ == "__main__":
    model = load_model({"model_choice":"unet", "in_channels":3}, "runs/try", "unet_pk")
    print(model)