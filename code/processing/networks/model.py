from code.utils import logging as log  # noqa: F401
from pathlib import Path

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def load(self, model_path: Path, device: str, model_name: str = "model.pt", **kwargs):
        self.load_state_dict(torch.load(model_path / model_name, map_location=device, **kwargs))
        self.to(device)

    def infer(self, data, device: str):
        self.eval()
        torch.set_grad_enabled(False)  # Disable gradient computation for testing
        return self(data.to(device)).detach()

    def save(self, path: Path, model_name: str = "model.pt"):
        torch.save(self.state_dict(), path / model_name)

        model_structure = []
        for name, param in self.named_parameters():
            model_structure.append([name, param.shape])
        with open(path / "model_structure.txt", "w") as f:
            f.write(str(model_structure))

    def num_of_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.zero_()
