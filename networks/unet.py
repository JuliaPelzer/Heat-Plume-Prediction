import torch.nn as nn
from torch import save, tensor, cat, load, equal
import pathlib

class UNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, init_features=32, depth=5, kernel_size=5, padding_mode="replicate", dilation = 2):
        super().__init__()
        features = init_features        
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        meta_model = []
        for _ in range(depth):
            self.encoders.append(UNet._block(in_channels, features, kernel_size=kernel_size, padding_mode=padding_mode, dilation = dilation))
            for i in range(3):
                meta_model.append({"kernel_size" : kernel_size,"stride":1, "dilation": dilation})
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            meta_model.append({"kernel_size" : 2,"stride":2, "dilation": 1})
            in_channels = features
            features *= 2
        self.encoders.append(UNet._block(in_channels, features, kernel_size=kernel_size, padding_mode=padding_mode, dilation = dilation))
        for i in range(3):
            meta_model.append({"kernel_size" : kernel_size,"stride":1, "dilation": dilation})
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for _ in range(depth):
            self.upconvs.append(nn.ConvTranspose2d(features, features//2, kernel_size=2, stride=2))
            self.decoders.append(UNet._block(features, features//2, kernel_size=kernel_size, padding_mode=padding_mode, dilation = dilation))
            features = features // 2

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)
        calc_receptive_field(meta_model)

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
    def _block(in_channels, features, kernel_size=5, padding_mode="zeros", dilation = 1):
        return nn.Sequential(
            # PaddingCircular(kernel_size, direction="both"),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=kernel_size,
                padding="same",
                padding_mode=padding_mode,
                dilation=dilation,
                bias=True,
            ),
            nn.ReLU(inplace=True),      
            # PaddingCircular(kernel_size, direction="both"),
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=kernel_size,
                padding="same",
                padding_mode=padding_mode,
                dilation=dilation,
                bias=True,
            ),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),      
            # PaddingCircular(kernel_size, direction="both"),
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=kernel_size,
                padding="same",
                padding_mode=padding_mode,
                dilation=dilation,
                bias=True,
            ),        
            nn.ReLU(inplace=True),
        )
    
    def load(self, model_path:pathlib.Path, device:str = "cpu", model_name: str = "model.pt", **kwargs):
        self.load_state_dict(load(model_path/model_name, **kwargs))
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

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.zero_()

class PaddingCircular(nn.Module):
    def __init__(self, kernel_size, direction="both"):
        super().__init__()
        self.pad_len = kernel_size//2
        self.direction = direction

    def forward(self, x:tensor) -> tensor:
        if self.direction == "both":
            padding_vector = (self.pad_len,)*4
            result = nn.functional.pad(x, padding_vector, mode='circular')    

        elif self.direction == "horizontal":
            padding_vector = (self.pad_len,)*2 + (0,)*2
            result = nn.functional.pad(x, padding_vector, mode='circular')    
            padding_vector = (0,)*2 + (self.pad_len,)*2  
            result = nn.functional.pad(result, padding_vector, mode='constant')     # or 'reflect'?
            
        return result

class UNetBC(UNet):
    def __init__(self, in_channels=2, out_channels=1, init_features=32, depth=3, kernel_size=5):
        super().__init__(in_channels, out_channels, init_features, depth, kernel_size)

        features = init_features
        for _ in range(depth): features *= 2
        for _ in range(depth): features = features // 2
        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    @staticmethod
    def _block(in_channels, features, kernel_size=5, padding_mode="zeros"):
        return nn.Sequential(
            ExplicitConv2dLayer(
                in_channels, features, kernel_size, bias=True,),
            nn.ReLU(inplace=True),    
            ExplicitConv2dLayer(
                features, features, kernel_size, bias=True,),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),  
            ExplicitConv2dLayer(
                features, features, kernel_size, bias=True,),
            nn.ReLU(inplace=True),
        )

def calc_receptive_field(meta_model):
    field_size = 1
    mult_stride = 1

    for i,curr_layer in enumerate(meta_model):
        field_size = field_size + ((curr_layer["kernel_size"]-1) * mult_stride)
        mult_stride = mult_stride * curr_layer["stride"]
    print(f"receptive field of current network is:{field_size}")