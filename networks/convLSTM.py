
# Based on the implementation by Rohit Panda (https://sladewinter.medium.com/video-frame-prediction-using-convlstm-network-in-pytorch-b5210a6ce582)

import torch
import torch.nn as nn
from torch import save, tensor, cat, load
import pathlib

# Original ConvLSTM cell as proposed by Shi et al. (after Rohit Panda)
class ConvLSTMCell(nn.Module):

    def __init__(self, in_channels, out_channels, activation, frame_size, conv_features, kernel_sizes):

        super(ConvLSTMCell, self).__init__()  

        if activation == "tanh":
            self.activation = torch.tanh 
        elif activation == "relu":
            self.activation = torch.relu
        elif activation == "sigmoid":
            self.activation = torch.sigmoid

        layers = []
        
        # Rohit Panda adapted this idea from https://github.com/ndrplz/ConvLSTM_pytorch
        layers.append(nn.Conv2d(
            in_channels=in_channels + out_channels, 
            out_channels=conv_features[0], 
            kernel_size=kernel_sizes[0], 
            stride=1,
            padding='same'
        ))

        layers.append(nn.ReLU(inplace=True))

        for i in range(1, len(conv_features)):
            layers.append(nn.Conv2d(
                in_channels=conv_features[i-1],
                out_channels=conv_features[i],
                kernel_size=kernel_sizes[i],
                stride=1,
                padding='same'
            ))
            layers.append(nn.BatchNorm2d(num_features=conv_features[i]))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(
            in_channels=conv_features[-1],
            out_channels=4 * out_channels,
            kernel_size=kernel_sizes[-1],
            stride=1,
            padding='same'
        ))

        layers.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*layers)

        # Initialize weights for Hadamard Products
        self.W_ci = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_co = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_cf = nn.Parameter(torch.Tensor(out_channels, *frame_size))

    def forward(self, X, H_prev, C_prev):

        # Rohit Panda adapted this idea from https://github.com/ndrplz/ConvLSTM_pytorch
        conv_output = self.conv[0](torch.cat([X, H_prev], dim=1))
        conv_output.to('cuda')
        
        for i in range(1, len(self.conv)):
            conv_output = self.conv[i](conv_output)

        # Rohit Panda adapted this idea from https://github.com/ndrplz/ConvLSTM_pytorch
        i_conv, f_conv, C_conv, o_conv = torch.chunk(conv_output, chunks=4, dim=1)

        input_gate = torch.sigmoid(i_conv + self.W_ci * C_prev )
        forget_gate = torch.sigmoid(f_conv + self.W_cf * C_prev )

        # Current Cell output
        C = forget_gate*C_prev + input_gate * self.activation(C_conv)

        output_gate = torch.sigmoid(o_conv + self.W_co * C )

        # Current Hidden State
        H = output_gate * self.activation(C)

        return H, C
    
    @staticmethod
    def _block(in_channels, features, kernel_size=5, padding_mode="same"):
        return nn.Sequential(
            
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=kernel_size,
                padding="same",
                # padding_mode=padding_mode,
                bias=True,
            ),
            nn.ReLU(inplace=True),      
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=kernel_size,
                padding="same",
                # padding_mode=padding_mode,
                bias=True,
            ),
            nn.BatchNorm3d(num_features=features),
            nn.ReLU(inplace=True),      
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=kernel_size,
                padding="same",
                bias=True,
            ),        
            nn.ReLU(inplace=True),
        )

class ConvLSTM(nn.Module):

    def __init__(self, in_channels, out_channels, activation, frame_size, prev_boxes, extend, 
                 conv_features, kernel_sizes):

        super(ConvLSTM, self).__init__()

        self.out_channels = out_channels
        self.prev_boxes = prev_boxes
        self.extend = extend

        # Initialize encoder cell and unroll
        self.convLSTMcell1 = ConvLSTMCell(in_channels, out_channels, activation, frame_size, conv_features,kernel_sizes)

        # Initialize decoder cell and unroll
        self.convLSTMcell2 = ConvLSTMCell(in_channels-1, out_channels, activation, frame_size, conv_features, kernel_sizes)

    def forward(self, X):

        # X is a frame sequence (batch_size, num_channels, seq_len, height, width)

        # Get the dimensions
        batch_size, _, seq_len, height, width = X.size()

        # Initialize output
        output = torch.zeros(batch_size, self.out_channels, seq_len,  height, width, device='cuda')
        
        # Initialize Hidden State
        H = torch.zeros(batch_size, self.out_channels, height, width, device='cuda')

        # Initialize Cell Input
        C = torch.zeros(batch_size,self.out_channels, height, width, device='cuda')

        # Unroll over time steps
        for time_step in range(self.prev_boxes):
            H, C = self.convLSTMcell1(X[:,:,time_step], H, C)

            output[:,:,time_step] = H

        for time_step in range(self.prev_boxes, self.prev_boxes+self.extend):
            H, C = self.convLSTMcell2(X[:,:-1,time_step], H, C)
            output[:,:,time_step] = H

        return output

class Seq2Seq(nn.Module):

    def __init__(self, in_channels, frame_size, prev_boxes, extend, num_layers,
    enc_conv_features,
    dec_conv_features,
    enc_kernel_sizes,
    dec_kernel_sizes,
    activation='relu'):
    
        super(Seq2Seq, self).__init__()

        assert enc_conv_features[-1] == dec_conv_features[0]

        self.sequential = nn.Sequential()
        self.prev_boxes = prev_boxes
        self.extend = extend

        # Add first ConvLSTM layer
        self.sequential.add_module(
            "convlstm1", ConvLSTM(
                in_channels=in_channels, out_channels=enc_conv_features[-1],  
                activation=activation, frame_size=frame_size, prev_boxes=prev_boxes, extend=extend,
                conv_features=enc_conv_features, kernel_sizes = enc_kernel_sizes)
        )

        self.sequential.add_module(
            "batchnorm1", nn.BatchNorm3d(num_features=enc_conv_features[-1])
        ) 

        # Add rest of the ConvLSTM layers
        for l in range(2, num_layers+1):

            self.sequential.add_module(
                f"convlstm{l}", ConvLSTM(
                    in_channels=enc_conv_features[-1], out_channels=enc_conv_features[-1],
                    activation=activation, frame_size=frame_size, prev_boxes=prev_boxes, extend=extend,
                    conv_features=enc_conv_features, kernel_sizes = enc_kernel_sizes)
                )
                
                
            self.sequential.add_module(
                f"batchnorm{l}", nn.BatchNorm3d(num_features=enc_conv_features[-1])
                ) 

        # Add convolution to predict output frame
        self.conv = nn.Sequential()

        self.conv.add_module("dec_conv1", nn.Conv2d(
            in_channels=dec_conv_features[0], 
            out_channels=dec_conv_features[1],
            kernel_size=dec_kernel_sizes[0], 
            stride=1,
            padding=(dec_kernel_sizes[0] - 1) // 2,  # Correct padding for 'same'
            bias=True
        ))
        self.conv.add_module("relu1", nn.ReLU(inplace=True))

        for i in range(1, len(dec_conv_features)-1):
            self.conv.add_module(f"dec_conv{i+1}", nn.Conv2d(
                in_channels=dec_conv_features[i],
                out_channels=dec_conv_features[i+1],
                kernel_size=dec_kernel_sizes[i],
                stride=1,
                padding=(dec_kernel_sizes[i] - 1) // 2,  # Correct padding for 'same'
                bias=True
            ))
            
            self.conv.add_module(f"batch{i}", nn.BatchNorm2d(num_features=dec_conv_features[i+1]))
            self.conv.add_module(f"relu{i+1}", nn.ReLU(inplace=True))

        self.conv.add_module("dec_conv_last",
            nn.Conv2d(
                in_channels=dec_conv_features[-1],
                out_channels=1,
                kernel_size=dec_kernel_sizes[-1],
                stride=1,
                padding=(dec_kernel_sizes[-1] - 1) // 2,  # Correct padding for 'same'
                bias=True
            )
        )       

        

    def forward(self, X):

        # Forward propagation through all the layers
        output = self.sequential(X)
        
        # get dimensions
        batch_size, _ , _, height, width = output.size()

        # initialize decoded output
        decoded_output = torch.zeros(batch_size, 1, self.extend, height, width, device='cuda')

        for pred_box in range(self.extend):
            curr_output = output[:,:,self.prev_boxes+pred_box]
            
            # decode current output
            decoded_output[:,:,pred_box] = self.conv(curr_output)
            
        decoded_output = torch.reshape(decoded_output, (decoded_output.shape[0], decoded_output.shape[1], width*self.extend, height))
        
        return nn.Sigmoid()(decoded_output)
    
    def save(self, path:pathlib.Path, model_name: str = "model.pt"):
        save(self.state_dict(), path/model_name)

        model_structure = []
        for name, param in self.named_parameters():
            model_structure.append([name, param.shape])
        with open(path/"model_structure.txt", "w") as f:
            f.write(str(model_structure))

    def load(self, model_path:pathlib.Path, device:str = "cpu", model_name: str = "model.pt"):
        self.load_state_dict(load(model_path/model_name))
        self.to(device)

    def num_of_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    
def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.zero_()