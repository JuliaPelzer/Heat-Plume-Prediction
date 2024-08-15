import torch
import numpy as np
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from itertools import product, repeat
import math

# function to rotate one datapoint counterclockwise (with pressure)
def rotate(data, angle):
    data_out = torch.zeros_like(data)
    # rotate all scalar fields
    for i in range(data.shape[0]):
        data_out[i] = TF.rotate(data[i].unsqueeze(0), angle).squeeze(0) #interpolation = InterpolationMode.BILINEAR
    
    return data_out

# rotate a datapoint such that direction matches specified direction and return rerotated inference (with pressure)
def rotate_and_infer(datapoint, grad_vec, model, info, device):
    p_ind = info['Inputs']['Liquid Pressure [Pa]']['index']
    center = int(datapoint[p_ind].shape[0]/2)
    angle = get_rotation_angle([datapoint[p_ind][center][center].item() - datapoint[p_ind][center + 1][center].item(), 
                                datapoint[p_ind][center][center].item() - datapoint[p_ind][center][center + 1].item()], grad_vec)
    x = rotate(datapoint, angle)

    #get inference
    x = x.to(device).unsqueeze(0)
    y_out = model(x).to(device)

    #rotate result back
    y_out = rotate(y_out.cpu().detach(), 360 - angle)
    return y_out

# rotate a datapoint such that direction matches specified direction and return rerotated inference (with pressure)
def rotate_and_infer_batch(batch, grad_vec, model, info, device):
    y_out_list = []
    p_ind = info['Inputs']['Liquid Pressure [Pa]']['index']
    center = int(batch[0][p_ind].shape[0]/2)
    for datapoint in batch:
        angle = get_rotation_angle([datapoint[p_ind][center][center].item() - datapoint[p_ind][center + 1][center].item(), 
                                    datapoint[p_ind][center][center].item() - datapoint[p_ind][center][center + 1].item()], grad_vec)
        x = rotate(datapoint, angle)

        #get inference
        x = x.to(device).unsqueeze(0)
        y_out = model(x).to(device)

        #rotate result back
        y_out = rotate(y_out.cpu().detach(), 360 - angle).squeeze(0)
        y_out_list.append(y_out)
    return torch.stack(y_out_list)

# get angle to rotate a counterclockwise to match b's direction
def get_rotation_angle(a,b):
    # calculate the dot product and the determinant
    dot_product = np.dot(a, b)
    determinant = a[0] * b[1] - a[1] * b[0]
    
    # calculate the angle in radians
    angle = np.degrees(np.arctan2(determinant, dot_product))
    
    # turn angle positive if necessary
    if angle < 0:
        angle += 360
    
    return angle

#build mask to cut out circular field from input, based on:
#https://quva-lab.github.io/escnn/api/escnn.nn.html?highlight=maskmodule#escnn.nn.MaskModule
def build_mask_gauss(
        s,
        dim: int = 2,
        margin: float = 0.0,
        sigma: float = 2.0,
        dtype=torch.float32,
):
    mask = torch.zeros(1, 1, *repeat(s, dim), dtype=dtype)
    c = (s-1) / 2
    t = (c - margin/100.*c)**2
    for k in product(range(s), repeat=dim):
        r = sum((x - c)**2 for x in k)
        if r > t:
            mask[(..., *k)] = math.exp((t - r) / sigma**2)
        else:
            mask[(..., *k)] = 1.
    return mask

#build mask to cut out circular field from input, based on:
#https://quva-lab.github.io/escnn/api/escnn.nn.html?highlight=maskmodule#escnn.nn.MaskModule
#returns tensor with just 0 and 1
def build_mask(
        s,
        dim: int = 2,
        dtype=torch.float32,
):
    mask = torch.zeros(1, 1, *repeat(s, dim), dtype=dtype)
    c = (s-1) / 2  # Center of the tensor
    r_max = c**2  # Maximum radius squared for the circle to fit

    for k in product(range(s), repeat=dim):
        r = sum((x - c)**2 for x in k)
        if r <= r_max:
            mask[(..., *k)] = 1.  # Inside the circle
        else:
            mask[(..., *k)] = 0.  # Outside the circle
    return mask

#cut out circular field of data
def mask_tensor(data):
    data_out = torch.zeros_like(data)
    mask = build_mask(data.shape[1], dtype = data.dtype)
    for i in range(data.shape[0]):
        data_out[i] = data[i]*mask
    
    return data_out

# FUNCTIONS TO USE IN COMBINATION WITH GRADIENT, ARTIFACTS OF EARLIER TINKERING ===> UNSUPPORTED!

# function to rotate one datapoint counterclockwise (with gradient)
def rotate_w_gradient(data, angle, info):
    x_grad_ind = y_grad_ind = -1
    data_out = torch.zeros_like(data)

    # for inputs change the gradient using rotation of the gradient vector
    if data.shape[0] > 1 in data:
        x_grad_ind = info['Inputs']['Pressure Gradient [-]']['index']
        y_grad_ind = info['Inputs']['Pressure Gradient [|]']['index']
        grad_x = data[x_grad_ind]
        grad_y = data[y_grad_ind]
        grad = np.array([grad_x[0][0],grad_y[0][0]])
    
        rad = np.radians(angle)
        rot_mat = np.array([[np.cos(rad), -np.sin(rad)],
                    [np.sin(rad), np.cos(rad)]])
        new_grad = np.dot(rot_mat, grad)
    
        data_out[x_grad_ind] = torch.full(grad_x.shape, new_grad[0])
        data_out[y_grad_ind] = torch.full(grad_y.shape, new_grad[1])

    # rotate all scalar fields
    for i in range(data.shape[0]):
        if i not in [x_grad_ind, y_grad_ind]:
            data_out[i] = TF.rotate(data[i].unsqueeze(0), angle).squeeze(0)
    
    return data_out

# rotate a datapoint such that direction matches specified direction and return rerotated inference (with gradient)
def rotate_and_infer_w_gradient(datapoint, grad_vec, model, info, device):
    # rotate datapoint
    x_grad_ind = info['Inputs']['Pressure Gradient [-]']['index']
    y_grad_ind = info['Inputs']['Pressure Gradient [|]']['index']
    angle = get_rotation_angle([datapoint[x_grad_ind][0][0].item(), datapoint[y_grad_ind][0][0].item()], grad_vec)
    x = rotate_w_gradient(datapoint, angle, info)

    #get inference
    x = x.to(device).unsqueeze(0)
    y_out = model(x).to(device)

    #rotate result back
    y_out = rotate_w_gradient(y_out.cpu().detach(), 360 - angle, info)
    return y_out