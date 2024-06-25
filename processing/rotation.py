import torch
import numpy as np
import torchvision.transforms.functional as TF

# function to rotate one datapoint counterclockwise
def rotate(data, angle, info):
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

# rotate a datapoint such that direction matches specified direction and return rerotated inference
def rotate_and_infer(datapoint, grad_vec, model, info, device):
    # rotate datapoint
    x_grad_ind = info['Inputs']['Pressure Gradient [-]']['index']
    y_grad_ind = info['Inputs']['Pressure Gradient [|]']['index']
    angle = get_rotation_angle([datapoint[x_grad_ind][0][0].item(), datapoint[y_grad_ind][0][0].item()], grad_vec)
    x = rotate(datapoint, angle, info)

    #get inference
    x = x.to(device).unsqueeze(0)
    y_out = model(x).to(device)

    #rotate result back
    y_out = rotate(y_out.cpu().detach(), 360 - angle, info)
    return y_out