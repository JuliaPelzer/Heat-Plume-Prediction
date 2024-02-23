import torch.nn as nn
from torch import save, tensor, cat, load, equal
import pathlib
import torch

class SVDPINN(nn.Module):
    def __init__(self, dataset, device, in_channels=4, out_channels=2, neurons_per_layer=80, hidden_layers=3, num_sing_values=80):
        super().__init__()    
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(num_sing_values+2, neurons_per_layer))
        self.layers.append(nn.ReLU())
        for _ in range(hidden_layers):
            self.layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(neurons_per_layer, num_sing_values))
        self.U = dataset.U[:, 0:num_sing_values].to(device)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x_in: tensor) -> tensor:
        x = x_in[:, 0:1, :, :]
        batch, dim, N, M = x.shape
        temp_in = torch.matmul(x.reshape(batch, 1,  N * M), self.U)[:, 0]
        permeability = x_in[:, 2, 0, 0].unsqueeze(1)
        gradient = x_in[:, 1, 0, 0].unsqueeze(1)
        
        x = torch.cat((temp_in, permeability, gradient), dim=1)
        for layer in self.layers:
            x = layer(x)

        x = torch.matmul(x.unsqueeze(1), torch.t(self.U)).reshape((batch, 1, N, M))
        #x = self.sigmoid(x)
        return torch.cat((x_in[:, 0:1, :, :], x), dim=2)

    def load(self, model_path:pathlib.Path, device:str = "cpu", model_name: str = "model.pt"):
        state_dict, self.U = load(model_path/model_name)
        self.load_state_dict(state_dict)
        self.to(device)

    def save(self, path:pathlib.Path, model_name: str = "model.pt"):
        model = (self.state_dict(), self.U)
        save(model, path/model_name)

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

class SVDPINN_one_extension(nn.Module):
    def __init__(self, dataset, device, number_iterations = 1, neurons_per_layer=80, hidden_layers=2, num_sing_values=50):
        super().__init__()    
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(num_sing_values+2, neurons_per_layer))
        self.layers.append(nn.ReLU())
        for _ in range(hidden_layers):
            self.layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(neurons_per_layer, num_sing_values))
        self.U = dataset.U[:, 0:num_sing_values].to(device)
        self.sigmoid = nn.Sigmoid()
        self.number_iterations = number_iterations


    def forward(self, x_in: tensor) -> tensor:
        x = x_in[:, 0:1, :, :]
        batch, dim, N, M = x.shape
        temp_in = torch.matmul(x.reshape(batch, 1,  N * M), self.U)[:, 0]
        permeability = x_in[:, 2, 0, 0].unsqueeze(1)
        gradient = x_in[:, 1, 0, 0].unsqueeze(1)
        output = [x]
        for i in range(self.number_iterations): #61
            x = torch.cat((temp_in, permeability, gradient), dim=1)
            for layer in self.layers:
                x = layer(x)
            temp_in = x
            x = torch.matmul(x.unsqueeze(1), torch.t(self.U)).reshape((batch, 1, N, M))
            output.append(x)
        #x = self.sigmoid(x)
        return torch.cat(output, dim=2)

    def load(self, model_path:pathlib.Path, device:str = "cpu", model_name: str = "model.pt"):
        state_dict, self.U = load(model_path/model_name)
        self.load_state_dict(state_dict)
        self.to(device)

    def save(self, path:pathlib.Path, model_name: str = "model.pt"):
        model = (self.state_dict(), self.U)
        save(model, path/model_name)

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