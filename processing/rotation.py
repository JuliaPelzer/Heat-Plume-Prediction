import torch
import numpy as np
import torchvision.transforms.functional as TF
import yaml

# function to read YAML file
def read_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

# function to rotate one datapoint counterclockwise
def rotate(data, angle, info_path):
    info = read_yaml(info_path)
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