from escnn import gspaces                                          #  1
from escnn import nn                                               #  2
import torch                                                       #  3
import torchvision.transforms.functional as TF

# function to rotate one datapoint counterclockwise (with pressure)
def rotate(data, angle):
    data_out = torch.zeros_like(data)
    # rotate all scalar fields
    for i in range(data.shape[0]):
        data_out[i] = TF.rotate(data[i].unsqueeze(0), angle).squeeze(0)
    
    return data_out
                                                                   #  4
r2_act = gspaces.rot2dOnR2(N=2)                                    #  5
feat_type_in  = nn.FieldType(r2_act,  3*[r2_act.trivial_repr])     #  6
feat_type_out = nn.FieldType(r2_act, 10*[r2_act.regular_repr])     #  7
                                                                   #  8
conv = nn.R2Conv(feat_type_in, feat_type_out, kernel_size=5)       #  9
relu = nn.ReLU(feat_type_out)                                      # 10
                                                                   # 11
x = torch.randn(1, 3, 32, 32)                                     # 12
x_rot = rotate(x.squeeze(0),180).unsqueeze(0)
x = feat_type_in(x)                                                # 13
x_rot = feat_type_in(x_rot)
                                                                   # 14
output = relu(conv(x)).tensor                                                  # 15
output_rot = rotate(relu(conv(x_rot)).tensor,180)



if torch.equal(output, output_rot):
    print('YEA')
else:
    print('NAY')

print(output)

print(output_rot)