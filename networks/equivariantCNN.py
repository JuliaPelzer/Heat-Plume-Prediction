from torch import save, tensor, cat, load, equal, randn

import torch
import torchvision.transforms.functional as TF

import torch.nn as nn
import pathlib
from escnn import gspaces
from escnn import nn as enn

class EquivariantCNN(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, init_features=32, depth=3, kernel_size=5, rotation_n=4):
        super().__init__()
        self.rotation_n = rotation_n
        self.gspace = gspaces.rot2dOnR2(N=rotation_n) #-1 for continuous
        features = init_features

        self.lifting_conv = enn.R2Conv(
            enn.FieldType(self.gspace, in_channels * [self.gspace.trivial_repr]), 
            enn.FieldType(self.gspace, in_channels * [self.gspace.regular_repr]), 
            kernel_size=1
        )
        
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        for _ in range(depth):
            self.encoders.append(self._block(in_channels, features, kernel_size=kernel_size))
            self.pools.append(enn.PointwiseMaxPool2D(enn.FieldType(self.gspace, features * [self.gspace.regular_repr]), kernel_size=2, stride=2))
            in_channels = features
            features *= 2
        self.encoders.append(self._block(in_channels, features, kernel_size=kernel_size))

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for _ in range(depth):
            self.upconvs.append(self._upsample_conv(features, features//2, kernel_size=3))     #Why does kernel_size=2 not work
            self.decoders.append(self._block(features, features//2, kernel_size=kernel_size))
            features = features // 2
        
        self.out_conv = enn.R2Conv(
            enn.FieldType(self.gspace, features * [self.gspace.regular_repr]), 
            enn.FieldType(self.gspace, out_channels * [self.gspace.trivial_repr]), 
            kernel_size=1
        )

    def _upsample_conv(self, in_channels, out_channels, kernel_size=3):
        in_type = enn.FieldType(self.gspace, in_channels * [self.gspace.regular_repr])
        out_type = enn.FieldType(self.gspace, out_channels * [self.gspace.regular_repr])
        return nn.Sequential(
            enn.R2Upsampling(in_type, scale_factor=2),
            enn.R2Conv(in_type, out_type, kernel_size=kernel_size, padding=kernel_size//2)
        )

    def _block(self, in_channels, features, kernel_size=5):
        in_type = enn.FieldType(self.gspace, in_channels * [self.gspace.regular_repr])
        out_type = enn.FieldType(self.gspace, features * [self.gspace.regular_repr])
        
        return nn.Sequential(
            enn.R2Conv(in_type, out_type, kernel_size, padding=kernel_size//2),
            enn.ReLU(out_type),
            enn.R2Conv(out_type, out_type, kernel_size, padding=kernel_size//2),
            enn.IIDBatchNorm2d(out_type),
            enn.ReLU(out_type),
            enn.R2Conv(out_type, out_type, kernel_size, padding=kernel_size//2),
            enn.ReLU(out_type)
        )

    def forward(self, x: tensor) -> tensor:
        x = enn.GeometricTensor(x, enn.FieldType(self.gspace, x.shape[1] * [self.gspace.trivial_repr]))
        x = self.lifting_conv(x)
        
        encodings = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            encodings.append(x)
            x = pool(x)
        x = self.encoders[-1](x)
        
        for upconv, decoder, encoding in zip(self.upconvs, self.decoders, reversed(encodings)):
            x = upconv(x)
            x = cat((x.tensor, encoding.tensor), dim=1)
            x = enn.GeometricTensor(x, enn.FieldType(self.gspace, x.shape[1]//self.rotation_n * [self.gspace.regular_repr])) # Problem with Unet structure? 
            x = decoder(x)
        
        return self.out_conv(x).tensor

    def load(self, model_path:pathlib.Path, device:str = "cpu", model_name: str = "model.pt"):
            self.load_state_dict(load(model_path/model_name))
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
        m.weights.data.normal_(0.0, 0.02)
    # elif classname.find("BatchNorm") != -1:
    #     m.weight.data.normal_(1.0, 0.02)
    #     m.bias.data.zero_()

# function to rotate one datapoint counterclockwise (with pressure)
# def rotate(data, angle):
#     data_out = torch.zeros_like(data)
#     # rotate all scalar fields
#     for i in range(data.shape[0]):
#         data_out[i] = TF.rotate(data[i].unsqueeze(0), angle).squeeze(0)
    
#     return data_out

# Test:
# in_channels = 1
# model = EquivariantCNN(in_channels=in_channels,rotation_n=2)
# input_tensor = randn(1, in_channels, 128, 128)
# rotated_input_tensor = rotate(input_tensor, 180)
# output = model(input_tensor)
# output_rot = rotate(model(rotated_input_tensor), 180)

# if torch.equal(output, output_rot):
#     print('YEA')
# else:
#     print('NAY')

# print(output.shape)
