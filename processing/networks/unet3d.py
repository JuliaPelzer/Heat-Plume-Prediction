import torch.nn as nn
from torch import save, tensor, cat, load, equal
import pathlib

class UNet3d(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, init_features=32, depth=3, kernel_size=5, activation_fct="ReLU"):
        super().__init__()
        features = init_features

        if activation_fct == "ReLU":
            activation = nn.ReLU(inplace=True)
        elif activation_fct == "LeakyReLU":
            activation = nn.LeakyReLU(inplace=True)

        padding_mode =  "zeros"            
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        for _ in range(depth):
            self.encoders.append(UNet3d._block(in_channels, features, kernel_size=kernel_size, padding_mode=padding_mode, activation_fct=activation))
            self.pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
            in_channels = features
            features *= 2
        self.encoders.append(UNet3d._block(in_channels, features, kernel_size=kernel_size, padding_mode=padding_mode, activation_fct=activation))

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for _ in range(depth):
            self.upconvs.append(nn.ConvTranspose3d(features, features//2, kernel_size=2, stride=2))
            self.decoders.append(UNet3d._block(features, features//2, kernel_size=kernel_size, padding_mode=padding_mode, activation_fct=activation))
            features = features // 2

        self.conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x: tensor) -> tensor:
        encodings = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            encodings.append(x)
            x = pool(x)
        x = self.encoders[-1](x)

        for upconv, decoder, encoding in zip(self.upconvs, self.decoders, reversed(encodings)):
            x = upconv(x)
            x = cat((x, encoding), dim=1)
            x = decoder(x)

        return self.conv(x)

    @staticmethod
    def _block(in_channels, features, kernel_size=5, padding_mode="zeros", activation_fct:nn.Module=nn.ReLU(inplace=True)):
        return nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=kernel_size,
                padding="same",
                bias=True,
            ),
            activation_fct,      
            nn.Conv3d(
                in_channels=features,
                out_channels=features,
                kernel_size=kernel_size,
                padding="same",
                bias=True,
            ),
            nn.BatchNorm3d(num_features=features),
            activation_fct,      
            nn.Conv3d(
                in_channels=features,
                out_channels=features,
                kernel_size=kernel_size,
                padding="same",
                bias=True,
            ),        
            activation_fct,
        )
    
    def load(self, model_path:pathlib.Path, device:str = "cpu", model_name: str = "model.pt"):
        self.load_state_dict(load(model_path/model_name))#,map_location="cpu"
        self.to(device)

    def save(self, path:pathlib.Path, model_name: str = "model.pt"):
        save(self.state_dict(), path/model_name)

        model_structure = []
        for name, param in self.named_parameters():
            model_structure.append([name, param.shape])
        with open(path/"model_structure.txt", "w") as f:
            f.write(str(model_structure))

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

def weights_init_3d(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        #nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        m.weight.data.normal_(0.0, 0.02)
        #m.weight.data.normal_(0.0, 0.004)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.zero_()

