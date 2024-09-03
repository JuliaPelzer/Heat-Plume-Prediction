import numpy as np
import torch 
import matplotlib.pyplot as plt
import pathlib
#from utils.utils_args import load_yaml
from networks.unetVariants import UNetHalfPad2

run_id = 977
inputs =  torch.load(f"/import/sgs.scratch/hofmanja/datasets_prepared/extend_plumes/ep_medium_1000dp_only_vary_dist inputs_gksi/Inputs/RUN_{run_id}.pt", map_location=torch.device('cpu'))
labels =  torch.load(f"/import/sgs.scratch/hofmanja/datasets_prepared/extend_plumes/ep_medium_1000dp_only_vary_dist inputs_gksi/Labels/RUN_{run_id}.pt", map_location=torch.device('cpu'))
model = UNetHalfPad2(5)
model.load(pathlib.Path("/home/hofmanja/1HP_NN/cnn model/ep_medium_2000dp_vary_values inputs_gksi box128 skip16"))

#info = load_yaml("/home/pelzerja/pelzerja/test_nn/1HP_NN/runs/extend/ep_medium_2000dp_vary_values inputs_gksi box128 skip16/info.yaml")
#cla = load_yaml("/home/pelzerja/pelzerja/test_nn/1HP_NN/runs/extend/ep_medium_2000dp_vary_values inputs_gksi box128 skip16/command_line_arguments.yaml")
print(inputs.shape, labels.shape)
#print(info)
#print(cla)

len_box = 128
skip = 32
overlap = 46
start = 0
outputs = torch.zeros_like(labels)
outputs[:,0:len_box+overlap] = labels[:,0:len_box+overlap]

for idx in range(0,30):
    input1 = inputs[:,start+len_box-overlap:start+2*len_box-overlap]
    output1 = labels[:,start:start+len_box]
    concat1 = torch.cat((input1, output1), 0)
    concat1 = concat1.unsqueeze(0)
    # print(concat1.shape)
    output1 = model(concat1)
    print(output1.shape)
    outputs[:,start+len_box+overlap:start+2*len_box-overlap] = output1[0,0,:,:].detach().cpu()
    # actual_len = len_box - 2*overlap
    # labels[:,start+len_box+overlap:start+len_box+overlap+actual_len] = output1[0,0,:,:].detach().cpu()
    start += skip

plt.figure(figsize=(20,2))
plt.imshow(outputs[0].T)
plt.colorbar()
plt.savefig("output.png")
plt.figure(figsize=(20,2))
plt.imshow(labels[0].T)
plt.colorbar()
plt.savefig("label.png")
plt.figure(figsize=(20,2))
plt.imshow((outputs[0]-labels[0]).T)
plt.colorbar()
plt.savefig("error.png")