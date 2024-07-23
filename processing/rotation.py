import torch
import numpy as np
import torchvision.transforms.functional as TF
from data_stuff.dataset import TrainDataset
from torch.utils.data import Subset

# function to rotate one datapoint counterclockwise (with pressure)
def rotate(data, angle):
    data_out = torch.zeros_like(data)
    # rotate all scalar fields
    for i in range(data.shape[0]):
        data_out[i] = TF.rotate(data[i].unsqueeze(0), angle).squeeze(0)
    
    return data_out

# rotate a datapoint such that direction matches specified direction and return rerotated inference (with pressure)
def rotate_and_infer(datapoint, grad_vec, model, info, device):
    p_ind = info['Inputs']['Liquid Pressure [Pa]']['index']
    angle = get_rotation_angle([datapoint[p_ind][1][1].item() - datapoint[p_ind][2][1].item(), datapoint[p_ind][1][1].item() - datapoint[p_ind][1][2].item()], grad_vec)
    x = rotate(datapoint, angle)

    #get inference
    x = x.to(device).unsqueeze(0)
    y_out = model(x).to(device)

    #rotate result back
    y_out = rotate(y_out.cpu().detach(), 360 - angle)
    return y_out

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

# augment data by adding rotated datapoints
def augment_data(dataset,  augmentation_n):
    inputs = [dataset[i][0] for i in range(len(dataset))]
    labels = [dataset[i][1] for i in range(len(dataset))]
    run_ids = [dataset.dataset.get_run_id(i) for i in range(len(dataset))]
    
    augmented_dataset = TrainDataset(dataset.dataset.path)

    for i in range(len(dataset)):
        augmented_dataset.add_item(inputs[i], labels[i], run_ids[i])

    
    for i in range(len(dataset)):
        for _ in range(augmentation_n):
            rot_angle = np.random.rand()*360
            #rot_angle = np.random.choice([0.25,0.5,0.75])*360
            augmented_dataset.add_item(rotate(inputs[i], rot_angle), rotate(labels[i], rot_angle), run_ids[i] + f'_rot_{rot_angle}')
        # for rot_angle in [90,180,270]:
        #     augmented_dataset.add_item(rotate(inputs[i], rot_angle), rotate(labels[i], rot_angle), run_ids[i] + f'_rot_{rot_angle}')

    return Subset(augmented_dataset, list(range(len(augmented_dataset))))


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