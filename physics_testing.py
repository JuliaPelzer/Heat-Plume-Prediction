import os
from data_stuff.dataset import SimulationDataset
import networks.losses as l
import torch
import matplotlib.pyplot as plt
import physics.equations_of_state as eq
import numpy as np


########## for testing the stencils

# def linear_x(x, y):
#     return x+5

# def linear_y(x, y):
#     return 2*y

# def bilinear(x, y):
#     return x*y

# def quadratic(x, y):
#     return x**2 + 3 * y**2

# def create_sample(x_list, y_list, f):
#     x_list.sort()
#     N = len(x_list)
#     y_list.sort()
#     M = len(y_list)
#     sample = torch.zeros((N,M))
#     for i in range(N):
#         for j in range(M):
#             sample[i,j] = f(x_list[i], y_list[j])
#     return sample

# x = torch.arange(0, 20, 2)
# y = torch.arange(-10, 10, 2)
# samples = torch.stack([create_sample(x, y, linear_x), create_sample(x, y, linear_y), create_sample(x, y, bilinear), create_sample(x, y, quadratic)])
# samples = samples.unsqueeze(1).to("cpu")

# loss = l.PhysicalLoss(device="cpu")

# dx = loss.central_differences_x(samples, 2)
# print(dx, dx.shape)

# dy = loss.central_differences_y(samples, 2)
# print(dy, dy.shape)

# lap = loss.laplace(samples, 2)
# print(lap, lap.shape)


###### for testing te loss at injection site and in general

def extract_sample_legacy(dataset, index, view):
    input, label = dataset[index]
    permeability = dataset.norm.reverse(input, "Inputs")[1]
    label = dataset.norm.reverse(label, "Labels")
    pressure = label[0]
    q_x = label[2] / 31536000
    q_y = label[1] / 31536000
    temperature = label[3]
    return permeability.view(1, 1, view[0], view[1]), pressure.view(1, 1, view[0], view[1]), q_x.view(1, 1, view[0], view[1]), q_y.view(1, 1, view[0], view[1]), temperature.view(1, 1, view[0], view[1])

def extract_sample(dataset, index, view):
    input, label = dataset[index]
    permeability = dataset.norm.reverse(input, "Inputs")[1]
    label = dataset.norm.reverse(label, "Labels")
    pressure = label[0]
    temperature = label[1]
    return permeability.view(1, 1, view[0], view[1]), pressure.view(1, 1, view[0], view[1]), temperature.view(1, 1, view[0], view[1])

def plot_sample(sample, name, i, variant="v1"):
    plt.imshow(sample.squeeze())
    plt.colorbar()
    v_min = torch.min(sample)
    v_max = torch.max(sample)
    plt.title("max_value: " + str(v_max.numpy()) + "  min_value: " + str(v_min.numpy()))
    plt.savefig("testing/sample_" + str(i) + "_" + name + "_" + variant + ".png", dpi=400)
    plt.close()

def descalarization(idx, shape):
    res = []
    N = np.prod(shape)
    for n in shape:
        N //= n
        res.append(idx // N)
        idx %= N
    return tuple(res)

dataset_name = "dataset_manuel_0_5_pksi" 
num__samples = 1

cell_width = 5
view = (256, 16)
faktor = 1
if dataset_name == "dataset_manuel_1_5_pksi":
    cell_width = 2.5
    view = (512, 32)
    faktor = 2
elif dataset_name == "dataset_manuel_2_5_pksi":
    cell_width = 1.25
    view = (1024, 64)
    faktor = 4


### calculate source terms

temperature = torch.tensor(15.6)
pressure = torch.tensor(910000)

inflow = torch.tensor(4.2) #m^3/day
inflow = inflow / 24 / 60 / 60 # convert to m^3/s
print("inflow: " + str(inflow.numpy()))
cell_volume = torch.tensor(cell_width) * cell_width * cell_width #m^3
_, molar_density = eq.eos_water_density_IFC67(temperature, pressure) # kmol/m^3
source = inflow / cell_volume * molar_density
print("source mass: " + str(source.numpy()))

source_energy = 76 * 1000 * source * (temperature - 10.6) * 2.5  #TODO fix this
# source_energy = source * (eq.eos_water_enthalphy(temperature, pressure) - pressure / molar_density)
print("source energy: " + str(source_energy.numpy()))

molar_density_func = lambda x: eq.eos_water_density_IFC67(temperature, x)
enthalpy_func = lambda x: eq.eos_water_enthalphy(temperature, x)
x = torch.arange(890000, 910000, 1000)
y_, y = molar_density_func(x)
y_ent = enthalpy_func(x)
plt.plot(x, y)
plt.savefig("testing/molar_density.png")
plt.close()

plt.plot(x, y_)
plt.savefig("testing/density.png")
plt.close()

plt.plot(x, y_ent)
plt.savefig("testing/enthalpy.png")
plt.close()


### plot v1

loss1 = l.PhysicalLoss(device="cpu")

dataset = SimulationDataset(os.path.join("/home/pillerls/heat-plume/datasets/prepared/legacy_datasets", dataset_name))
for i in range(num__samples):
    permeability, pressure, q_x, q_y, temperature = extract_sample_legacy(dataset, i, view)

    plot_sample(permeability, "permeability", i)
    plot_sample(pressure, "pressure", i)
    plot_sample(q_x, "q_x", i)
    plot_sample(q_y, "q_y", i)
    plot_sample(temperature, "temperature", i)

    continuity_error = loss1.get_continuity_error(temperature, pressure, q_x, q_y, cell_width)
    darcy_x_error, darcy_y_error = loss1.get_darcy_errors(temperature, pressure, q_x, q_y, permeability, cell_width)
    energy_error = loss1.get_energy_error(temperature, pressure, q_x, q_y, cell_width)
    plot_sample(continuity_error[:, :, 30*faktor:,:], "continuity_error_lower", i)
    plot_sample(continuity_error, "continuity_error", i)
    plot_sample(darcy_x_error, "darcy_x_error", i)
    plot_sample(darcy_y_error, "darcy_y_error", i)
    plot_sample(energy_error[:, :, 30*faktor:,:], "energy_error_lower", i)
    plot_sample(energy_error, "energy_error", i)

    index = torch.argmax(continuity_error, )
    continuity_error[descalarization(index, continuity_error.shape)] -= source
    index = torch.argmax(energy_error)
    energy_error[descalarization(index, energy_error.shape)] -= source_energy
    plot_sample(continuity_error, "continuity_error_minus", i)
    plot_sample(energy_error, "energy_error_minus", i)


### plot v2

loss2 = l.PhysicalLossV2(device="cpu")

for i in range(num__samples):
    permeability, pressure, _, _, temperature = extract_sample_legacy(dataset, i, view)
    q_x, q_y = loss2.get_darcy(temperature, pressure, permeability, cell_width)

    plot_sample(q_x, "q_x", i, "v2")
    plot_sample(q_y, "q_y", i, "v2")

    continuity_error = loss2.get_continuity_error(temperature, pressure,permeability,  cell_width)
    energy_error = loss2.get_energy_error(temperature, pressure, permeability, cell_width)

    plot_sample(continuity_error[:, :, 30*faktor:,:], "continuity_error_lower", i, "v2")
    plot_sample(continuity_error, "continuity_error", i, "v2")
    plot_sample(darcy_x_error, "darcy_x_error", i, "v2")
    plot_sample(darcy_y_error, "darcy_y_error", i, "v2")
    plot_sample(energy_error[:, :, 30*faktor:,:], "energy_error_lower", i, "v2")
    plot_sample(energy_error, "energy_error", i, "v2")

    index = descalarization(torch.argmax(energy_error), energy_error.shape)
    continuity_error[index] -= source
    energy_error[index] -= source_energy
    plot_sample(continuity_error, "continuity_error_minus", i, "v2")
    plot_sample(energy_error, "energy_error_minus", i, "v2")
    print("continuity loss: ", torch.mean(torch.pow(continuity_error, 2)))
    print("energy loss: ", torch.mean(torch.pow(energy_error, 2)))
    continuity_error = loss2.get_continuity_error(torch.zeros_like(temperature), torch.zeros_like(pressure),permeability,  cell_width)
    energy_error = loss2.get_energy_error(torch.zeros_like(temperature), torch.zeros_like(pressure), permeability, cell_width)
    continuity_error[index] -= source
    energy_error[index] -= source_energy
    print("continuity loss to zero: ", torch.mean(torch.pow(continuity_error, 2)))
    print("energy loss to zero: ", torch.mean(torch.pow(energy_error, 2)))



### real dataset
if dataset_name == "dataset_manuel_0_5_pksi":
    dataset = SimulationDataset("/home/pillerls/heat-plume/datasets/prepared/benchmark_dataset_2d_10datapoints_pksi")
    for i in range(num__samples):
        permeability, pressure, temperature = extract_sample(dataset, i, view)
        q_x, q_y = loss2.get_darcy(temperature, pressure, permeability, cell_width)

        plot_sample(q_x, "q_x", i, "v3")
        plot_sample(q_y, "q_y", i, "v3")

        continuity_error = loss2.get_continuity_error(temperature, pressure,permeability,  cell_width)
        energy_error = loss2.get_energy_error(temperature, pressure, permeability, cell_width)

        plot_sample(continuity_error[:, :, 30*faktor:,:], "continuity_error_lower", i, "v3")
        plot_sample(continuity_error, "continuity_error", i, "v3")
        plot_sample(darcy_x_error, "darcy_x_error", i, "v3")
        plot_sample(darcy_y_error, "darcy_y_error", i, "v3")
        plot_sample(energy_error[:, :, 30*faktor:,:], "energy_error_lower", i, "v3")
        plot_sample(energy_error, "energy_error", i, "v3")

        index = descalarization(torch.argmax(energy_error), energy_error.shape)
        continuity_error[index] -= source
        energy_error[index] -= source_energy
        plot_sample(continuity_error, "continuity_error_minus", i, "v3")
        plot_sample(energy_error, "energy_error_minus", i, "v3")
        print("continuity loss: ", torch.mean(torch.pow(continuity_error, 2)))
        print("energy loss: ", torch.mean(torch.pow(energy_error, 2)))
        continuity_error = loss2.get_continuity_error(torch.zeros_like(temperature), torch.zeros_like(pressure),permeability,  cell_width)
        energy_error = loss2.get_energy_error(torch.zeros_like(temperature), torch.zeros_like(pressure), permeability, cell_width)
        continuity_error[index] -= source
        energy_error[index] -= source_energy
        print("continuity loss to zero: ", torch.mean(torch.pow(continuity_error, 2)))
        print("energy loss to zero: ", torch.mean(torch.pow(energy_error, 2)))