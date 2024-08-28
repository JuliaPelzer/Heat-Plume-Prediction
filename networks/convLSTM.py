
import torch
import torch.nn as nn
from torch import save, tensor, cat, load
import pathlib

# Original ConvLSTM cell as proposed by Shi et al.
class ConvLSTMCell(nn.Module):

    def __init__(self, in_channels, out_channels, activation, frame_size):

        super(ConvLSTMCell, self).__init__()  

        if activation == "tanh":
            self.activation = torch.tanh 
        elif activation == "relu":
            self.activation = torch.relu
        
        self.nr_features = [16, 32, 64, 64, out_channels]
        self.kernel_sizes = [7, 5, 5, 5, 5]

        layers = []
        
        layers.append(nn.Conv2d(
            in_channels=in_channels + out_channels, 
            out_channels=self.nr_features[0], 
            kernel_size=self.kernel_sizes[0], 
            stride=1,
            padding='same'
        ))

        layers.append(nn.ReLU(inplace=True))

        for i in range(1, len(self.nr_features) - 1):
            # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
            layers.append(nn.Conv2d(
                in_channels=self.nr_features[i-1],
                out_channels=self.nr_features[i],
                kernel_size=self.kernel_sizes[i],
                stride=1,
                padding='same'
            ))
            layers.append(nn.BatchNorm2d(num_features=self.nr_features[i]))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(
            in_channels=self.nr_features[-1],
            out_channels=4 * out_channels,
            kernel_size=self.kernel_sizes[-1],
            stride=1,
            padding='same'
        ))

        layers.append(nn.ReLU(inplace=True))

        # Create a Sequential container with the layers
        self.conv = nn.Sequential(*layers)

        # Initialize weights for Hadamard Products
        self.W_ci = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_co = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_cf = nn.Parameter(torch.Tensor(out_channels, *frame_size))

    def forward(self, X, H_prev, C_prev):

        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch   
        assert len(X.size()) == 4, f'expected 4 but got {X.size()}'
        assert len(H_prev.size()) == 4, f'expected 4 but got {H_prev.size()}'
        conv_output = self.conv[0](torch.cat([X, H_prev], dim=1))
        conv_output.to('cuda')
        
        for i in range(1, len(self.conv)):
            conv_output = self.conv[i](conv_output)

        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        i_conv, f_conv, C_conv, o_conv = torch.chunk(conv_output, chunks=4, dim=1)

        with open('/home/hofmanja/1HP_NN/test4.txt', 'w') as file:
            file.write(f'i_conv shape: {i_conv.shape}\n')
            file.write(f'self.W_ci: {self.W_ci.shape}\n')
            file.write(f'C_prev: {C_prev.shape}\n' )

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
            # PaddingCircular(kernel_size, direction="both"),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=kernel_size,
                padding="same",
                # padding_mode=padding_mode,
                bias=True,
            ),
            nn.ReLU(inplace=True),      
            # PaddingCircular(kernel_size, direction="both"),
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
            # PaddingCircular(kernel_size, direction="both"),
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=kernel_size,
                padding="same",
                # padding_mode=padding_mode,
                bias=True,
            ),        
            nn.ReLU(inplace=True),
        )

class ConvLSTM(nn.Module):

    def __init__(self, in_channels, out_channels, activation, frame_size, prev_boxes, extend):

        super(ConvLSTM, self).__init__()

        self.out_channels = out_channels
        self.prev_boxes = prev_boxes
        self.extend = extend

        # We will unroll this over time steps
        self.convLSTMcell1 = ConvLSTMCell(in_channels, out_channels, activation, frame_size)

        # initialize last cel
        self.convLSTMcell2 = ConvLSTMCell(in_channels-1, out_channels, activation, frame_size)

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

    def __init__(self, num_channels, frame_size, prev_boxes, extend, num_layers, num_kernels=64,
    activation='relu'):

        super(Seq2Seq, self).__init__()

        self.sequential = nn.Sequential()
        self.prev_boxes = prev_boxes
        self.extend = extend

        # Add First layer (Different in_channels than the rest)
        self.sequential.add_module(
            "convlstm1", ConvLSTM(
                in_channels=num_channels, out_channels=num_kernels,  
                activation=activation, frame_size=frame_size, prev_boxes=prev_boxes, extend=extend)
        )

        self.sequential.add_module(
            "batchnorm1", nn.BatchNorm3d(num_features=num_kernels)
        ) 

        # Add rest of the layers
        for l in range(2, num_layers+1):

            self.sequential.add_module(
                f"convlstm{l}", ConvLSTM(
                    in_channels=num_kernels, out_channels=num_kernels,
                    activation=activation, frame_size=frame_size, prev_boxes=prev_boxes, extend=extend)
                )
                
                
            self.sequential.add_module(
                f"batchnorm{l}", nn.BatchNorm3d(num_features=num_kernels)
                ) 

        # Add Convolutional Layer to predict output frame
        features = [64, 64, 64]
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=num_kernels, 
                out_channels=features[0],
                kernel_size=5, 
                stride=1,
                padding='same',
                bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=features[0], 
                out_channels=features[1],
                kernel_size=5, 
                stride=1, 
                padding='same',
                bias=True),
            nn.BatchNorm2d(num_features=features[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=features[2],
                out_channels=1,
                kernel_size=5,
                padding='same',
                bias=True)
            )

    def forward(self, X):

        # Forward propagation through all the layers
        output = self.sequential(X)


        # decode result
        batch_size, _ , _, height, width = output.size()
        new_output = torch.zeros(batch_size, 1, self.extend, height, width, device='cuda')

        for pred_box in range(self.extend):
            curr_output = output[:,:,self.prev_boxes+pred_box]
            #assert curr_output.shape == [50, 64, 64, 64], f'got {curr_output.shape}'
            new_output[:,:,pred_box] = self.conv(output[:,:,self.prev_boxes+pred_box])
            
        new_output = torch.reshape(new_output, (new_output.shape[0], new_output.shape[1], width*self.extend, height))
        
        return nn.Sigmoid()(new_output)
    
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

    
def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.zero_()