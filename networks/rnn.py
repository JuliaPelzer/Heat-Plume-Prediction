import torch.nn as nn
from torch import save, tensor, cat, load, equal
import pathlib
import torch

class RNNPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(2, 2*64*64, batch_first=True)


    def forward(self, x: tensor) -> tensor:
        batch, _, N, M = x.shape
        temp_in = torch.matmul(x[:, 0:1, :, :].reshape(batch, 1, N * M), self.U)[:, 0]
        permeability = x[:, 2, 0, 0].unsqueeze(1)
        gradient = x[:, 3, 0, 0].unsqueeze(1)
        
        x = torch.cat((temp_in, permeability, gradient), dim=1)
        for layer in self.layers:
            x = layer(x)

        x = torch.matmul(x.unsqueeze(1), torch.t(self.U))
        return x.reshape((batch, 1, N, M))

    def load(self, model_path:pathlib.Path, device:str = "cpu", model_name: str = "model.pt"):
        self.load_state_dict(load(model_path/model_name))
        self.to(device)

    def save(self, path:pathlib.Path, model_name: str = "model.pt"):
        save(self.state_dict(), path/model_name)

    def num_of_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def compare(self, model_2):
        # source: https://discuss.pytorch.org/t/check-if-models-have-same-weights/4351/3
        try:
            # can handle both: model2 being only a state_dict or a full model
            model_2 = model_2.state_dict()
        except:
            pass    
        models_differ = 0
        for key_item_1, key_item_2 in zip(self.state_dict().items(), model_2.items()):
            if equal(key_item_1[1], key_item_2[1]):
                pass
            else:
                models_differ += 1
                if (key_item_1[0] == key_item_2[0]):
                    print('Mismatch found at', key_item_1[0])
                else:
                    raise Exception
        if models_differ == 0:  print('Models match perfectly! :)')


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2*64*128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 40),
        )

        self.decoder = nn.Sequential(
            nn.Linear(40, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 2*64*128),
        )


    def forward(self, x: tensor) -> tensor:
        x = x[:, 0:2]
        orig_shape = x.shape
        x = x.reshape((orig_shape[0], orig_shape[1]*orig_shape[2]*orig_shape[3]))
        
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.reshape(orig_shape)
        return x

    def load(self, model_path:pathlib.Path, device:str = "cpu", model_name: str = "model.pt"):
        self.load_state_dict(load(model_path/model_name))
        self.to(device)

    def save(self, path:pathlib.Path, model_name: str = "model.pt"):
        save(self.state_dict(), path/model_name)

    def num_of_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def compare(self, model_2):
        # source: https://discuss.pytorch.org/t/check-if-models-have-same-weights/4351/3
        try:
            # can handle both: model2 being only a state_dict or a full model
            model_2 = model_2.state_dict()
        except:
            pass    
        models_differ = 0
        for key_item_1, key_item_2 in zip(self.state_dict().items(), model_2.items()):
            if equal(key_item_1[1], key_item_2[1]):
                pass
            else:
                models_differ += 1
                if (key_item_1[0] == key_item_2[0]):
                    print('Mismatch found at', key_item_1[0])
                else:
                    raise Exception
        if models_differ == 0:  print('Models match perfectly! :)')

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.zero_()