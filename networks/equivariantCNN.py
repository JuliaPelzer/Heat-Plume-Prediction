from torch import save, load, equal, randn, tensor

import torch.nn as nn
import pathlib
from escnn import gspaces
from escnn import nn as enn

class G_UNet(nn.Module):
    def __init__(self, in_channels : int =2, out_channels : int =1, init_features : int=32, depth : int=3, kernel_size : int=5, rotation_n : int=4):
        """
        Creates an equivariant UNet.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            init_features (int): Number of features in the first layer (later divided by rotation_n).
            depth (int): Number of UNet blocks in the encoder and decoder sides.
            kernel_size (int): Kernel size for convolutional layers.
            rotation_n (int): Number of rotations denoting the cyclic group Cn used.
        """
        super(G_UNet, self).__init__()
    
        # mandatory input
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = [int(init_features/rotation_n*pow(2,i)) for i in range(0,depth)]
        self.kernel_size = kernel_size

        # set the model equivariance under rotations by specified cyclic group
        self.r2_act = gspaces.rot2dOnR2(N=rotation_n)
         
        # set input/output type as trivial representation
        self.input_type = enn.FieldType(self.r2_act, self.in_channels*[self.r2_act.trivial_repr])
        self.output_type = enn.FieldType(self.r2_act, self.out_channels*[self.r2_act.trivial_repr])        
            
        # create list with encoder channels
        enc_dims = [self.in_channels, *self.channels]
        
        # convert to rt_act's
        enc_types = self.convert_to_r2_act(enc_dims)
        enc_types[0] = self.input_type # manually convert to trivial_repr because input
        
        self.enc_conv = nn.ModuleList()
        self.enc_pool = nn.ModuleList()
        
        # create encoders
        for i in range(1, len(enc_dims)):
            self.enc_conv.append(self._block(enc_types[i-1], enc_types[i], kernel_size=self.kernel_size))
            
            # create pooling seperatly
            self.enc_pool.append(enn.PointwiseMaxPool2D(enc_types[i], kernel_size=2, stride=2))           
            
        # create bottleneck
        mid_type = enn.FieldType(self.r2_act, 2*enc_dims[-1]*[self.r2_act.regular_repr])
        self.bottleneck = self._block(enc_types[-1], mid_type, kernel_size=self.kernel_size)

        # create list with decoder channels
        dec_dims = [*self.channels, 2*self.channels[-1]]
        dec_dims.reverse()
        
        # convert to rt_act's
        dec_types = self.convert_to_r2_act(dec_dims)
                
        self.dec_conv = nn.ModuleList()
        self.dec_upsample = nn.ModuleList()

        # create decoder
        for i in range(1, len(dec_dims)):
            self.dec_conv.append(self._block(dec_types[i-1], dec_types[i], kernel_size=self.kernel_size))
            
            # create upsampling seperatly
            self.dec_upsample.append(self._upsample_conv(dec_types[i-1], dec_types[i], kernel_size=3))
            
        # create tail end
        self.tail_conv = enn.SequentialModule(*self.get_layer(dec_types[-1], self.output_type, kernel_size=1, bn=False, activation=None))

    def get_layer(self, in_type : enn.FieldType, out_type : enn.FieldType, kernel_size : int = 5,
                  bn : bool = False, activation : str = 'relu') -> list:
        """
        Creates a convolutional layer based on specified specifications.

        Args:
            in_type (FieldType): Type of input data.
            out_type (FieldType): Type of output data.
            kernel_size (int): Size of the convolution kernel.
            bn (bool): Whether to add batch normalization after convolution.
            activation (str): Activation function to apply after convolution.
                                        Currently, only 'relu' is supported.

        Returns:
            list: List representing convolutional layer.
        """
        out_list = []
    
        # add convolutional layer
        out_list.append(
            enn.R2Conv(in_type, out_type, kernel_size=kernel_size, padding=kernel_size//2, bias=True)
        )
    
        # add batch norm
        if bn:
            out_list.append(enn.InnerBatchNorm(out_type))
    
        # add activation
        if activation is None:
            pass
        elif activation == 'relu':
            out_list.append(enn.ReLU(out_type, inplace=True))
        else:
            raise Exception('Specified activation function not implemented!')
          
        return out_list
    
    # build a UNet block consisting of 3 convolutional layers with ReLU and a single batch norm 
    def _block(self, in_type : enn.FieldType, out_type : enn.FieldType, kernel_size :int = 5) -> enn.SequentialModule:
        return enn.SequentialModule(
            *self.get_layer(in_type, out_type, kernel_size=kernel_size),
            *self.get_layer(out_type, out_type, bn=True, kernel_size=kernel_size),
            *self.get_layer(out_type, out_type, kernel_size=kernel_size)
        )
    
    def _upsample_conv(self, in_type : enn.FieldType, out_type : enn.FieldType, kernel_size : int = 3) -> enn.SequentialModule:
        return enn.SequentialModule(
            enn.R2Upsampling(in_type, scale_factor=2),
            enn.R2Conv(in_type, out_type, kernel_size=kernel_size, padding=kernel_size//2)
        )
    
    def convert_to_r2_act(self,dims : list) -> list:
        """
        Builds a list of types encoding how group actions act on the data and its dimensionality.

        Args:
            dims (list of int): List of dimensions to create corresponding types for.

        Returns:
            list: List of types representing group actions on the data based on the given dimensions.
        """
        types= []
        for dim in dims:
            types.append(enn.FieldType(self.r2_act, dim*[self.r2_act.regular_repr]))
        return types

    def forward(self, x : tensor) -> tensor:
        # convert to geometric tensor
        x = enn.GeometricTensor(x, self.input_type)        
        
        enc = x # x is input of first encoder
        
        # pass through the encoder
        enc_out = []
        for i in range(len(self.enc_conv)):
            # pass through a single encoder layer
            enc = self.enc_conv[i](enc)

            # save the encoder output such that it can be used for skip connections
            enc_out.append(enc)

            # downsample with convolutional pooling
            enc = self.enc_pool[i](enc)

        # pass through the bottleneck
        b = self.bottleneck(enc)

        # pass through the decoder
        dec = b
        enc_out.reverse()   # reverse such that it fits pass through decoder
        for i in range(len(self.dec_conv)):
            # get input for decoder
            dec = self.dec_upsample[i](dec)
            # perform skip connections
            dec = enn.tensor_directsum([enc_out[i],dec])

            # pass through a single decoder layer
            dec = self.dec_conv[i](dec)
            
        # perform tail convolutions
        x = self.tail_conv(dec) # no activation
        x = x.tensor
        return x

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
