import os
from data_stuff.dataset import SimulationDataset
import networks.losses as l
import torch

def linear_x(x, y):
    return x+5

def linear_y(x, y):
    return 2*y

def bilinear(x, y):
    return x*y

def quadratic(x, y):
    return x**2 + 3 * y**2

def create_sample(x_list, y_list, f):
    x_list.sort()
    N = len(x_list)
    y_list.sort()
    M = len(y_list)
    sample = torch.zeros((N,M))
    for i in range(N):
        for j in range(M):
            sample[i,j] = f(x_list[i], y_list[j])
    return sample

# x = torch.arange(0, 20, 2)
# y = torch.arange(-10, 10, 2)
# samples = torch.stack([create_sample(x, y, linear_x), create_sample(x, y, linear_y), create_sample(x, y, bilinear), create_sample(x, y, quadratic)])
# samples = samples.unsqueeze(1).to("cpu")

loss = l.PhysicalLoss(device="cpu")

# dx = loss.central_differences_x(samples, 2)
# print(dx, dx.shape)

# dy = loss.central_differences_y(samples, 2)
# print(dy, dy.shape)

# lap = loss.laplace(samples, 2)
# print(lap, lap.shape)



dataset_name = "try_velocity_pksi"

dataset = SimulationDataset(os.path.join("/home/pillerls/heat-plume/datasets/prepared", dataset_name))

print(dataset[0])