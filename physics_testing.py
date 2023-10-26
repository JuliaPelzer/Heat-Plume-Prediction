import os
from data_stuff.dataset import SimulationDataset
import networks.losses as l
import torch
import matplotlib.pyplot as plt
import physics.equations_of_state as eq
import numpy as np
from utils.visualize_data import _aligned_colorbar


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
    inputs = dataset.norm.reverse(input, "Inputs")
    permeability = inputs[1]
    pressure_start = inputs[0]
    label = dataset.norm.reverse(label, "Labels")
    pressure = label[0]
    temperature = label[1]
    return permeability.view(1, 1, view[0], view[1]), pressure.view(1, 1, view[0], view[1]), temperature.view(1, 1, view[0], view[1]), pressure_start.view(1, 1, view[0], view[1])


def plot_sample(sample, name, i, variant="v1", sources = None):
    plt.imshow(torch.t(sample.squeeze()))
    plt.colorbar()
    #_aligned_colorbar()
    v_min = torch.min(sample)
    v_max = torch.max(sample)
    plt.title("max_value: " + str(v_max.numpy()) + "  min_value: " + str(v_min.numpy()))
    if not sources == None:
        plt.suptitle("source_cont: " + str(sources[0]) + "   source_energy: " + str(sources[1]))
    plt.savefig("testing/" + str(i) + "_" + name + "_" + variant + ".png", dpi=400)
    plt.close()

def descalarization(idx, shape):
    res = []
    N = np.prod(shape)
    for n in shape:
        N //= n
        res.append(idx // N)
        idx %= N
    return tuple(res)

legacy = False

temperature = torch.tensor(10.6)
pressure = torch.tensor(910000)
molar_density_func = lambda x: eq.eos_water_density_IFC67(x, pressure)
enthalpy_func = lambda x: eq.eos_water_enthalphy(x, pressure)
x = torch.arange(8, 17, 0.1)
y_, y = molar_density_func(x)
y_ent = enthalpy_func(x)
plt.plot(x, y)
plt.savefig("testing/molar_density_temp.png")
plt.close()

plt.plot(x, y_)
plt.savefig("testing/density_temp.png")
plt.close()

plt.plot(x, y_ent)
plt.savefig("testing/enthalpy_temp.png")
plt.close()

molar_density_func = lambda x: eq.eos_water_density_IFC67(temperature, x)
enthalpy_func = lambda x: eq.eos_water_enthalphy(temperature, x)
x = torch.arange(890000, 920000, 10000)
y_, y = molar_density_func(x)
y_ent = enthalpy_func(x)
plt.plot(x, y)
plt.savefig("testing/molar_density_press.png")
plt.close()

plt.plot(x, y_)
plt.savefig("testing/density_press.png")
plt.close()

plt.plot(x, y_ent)
plt.savefig("testing/enthalpy_press.png")
plt.close()

if legacy:
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


def create_example_plots(dataset, dataset_name, temp, flux):
    print("iteration: " + str(temp) + " and " + str(flux))
    index = (0, 0, 22, 6) # get position of heat pump (fixed TODO read in)
    loss2 = l.PhysicalLossV2(device="cpu")
    cell_width = 5
    view = (256, 16)
    permeability, pressure, temperature, pressure_start = extract_sample(dataset, 0, view)
    temperature_inflow = torch.tensor(temp + 10.6)
    pressure_at_hp = pressure[index]
    inflow = torch.tensor(0.00024) * flux # convert to m^3/s
    print("inflow: " + str(inflow.numpy()))
    cell_volume = torch.tensor(cell_width) * cell_width * cell_width #m^3
    _, molar_density = eq.eos_water_density_IFC67(temperature_inflow, pressure_at_hp) # kmol/m^3
    source = inflow / cell_volume * molar_density
    print("source mass: " + str(source.numpy()))
    ###weird test
    source_energy = source * (eq.eos_water_enthalphy(temperature_inflow, pressure_at_hp) - pressure_at_hp / molar_density / 1000)
    ###weird test
    print("source energy: " + str(source_energy.numpy()))
    sources = (source, source_energy)

    q_x, q_y = loss2.get_darcy(temperature, pressure, permeability, cell_width)
    
    plot_sample(pressure, "pressure", 0, dataset_name, sources)
    plot_sample(temperature, "temperature", 0, dataset_name, sources)
    # plot_sample(q_x, "q_x", 0, dataset_name, sources)
    # plot_sample(q_y, "q_y", 0, dataset_name, sources)

    continuity_error = loss2.get_continuity_error(temperature, pressure, permeability,  cell_width)
    energy_error = loss2.get_energy_error(temperature, pressure, permeability, cell_width)

    plot_sample(continuity_error[:, :, 30:,:], "continuity_error_lower", 0, dataset_name, sources)
    plot_sample(continuity_error, "continuity_error", 0, dataset_name, sources)
    plot_sample(energy_error[:, :, 30:,:], "energy_error_lower", 0, dataset_name, sources)
    plot_sample(energy_error, "energy_error", 0, dataset_name, sources)

    continuity_error[index] -= source
    energy_error[index] -= source_energy
    plot_sample(continuity_error, "continuity_error_minus", 0, dataset_name, sources)
    plot_sample(energy_error, "energy_error_minus", 0, dataset_name, sources)
    print("continuity loss: ", torch.mean(torch.pow(continuity_error, 2)))
    print("energy loss: ", torch.mean(torch.pow(energy_error, 2)))
    continuity_error = loss2.get_continuity_error(torch.zeros_like(temperature), torch.zeros_like(pressure),permeability,  cell_width)
    energy_error = loss2.get_energy_error(torch.zeros_like(temperature), torch.zeros_like(pressure), permeability, cell_width)
    continuity_error[index] -= source
    energy_error[index] -= source_energy
    print("continuity loss to zero: ", torch.mean(torch.pow(continuity_error, 2)))
    print("energy loss to zero: ", torch.mean(torch.pow(energy_error, 2)))


    plot_sample(pressure - pressure_start, "pressure_delta", 0, dataset_name, sources)

temps = [0, 5, 10, -5, -10]
fluxs = [0.5, 1, 2]
for temp in temps:
    for flux in fluxs:
        temp_str = str(temp)
        flux_str = str(flux)
        if flux == 0.5: flux_str="0_5"
        dataset_name = "lukas_T"+temp_str+"_flux"+flux_str+"_pksi"
        ### real dataset
        if not legacy:
            dataset = SimulationDataset("/home/pillerls/heat-plume/datasets/prepared/dataset_lukas_fixed_perm_p/"+dataset_name)
            create_example_plots(dataset, dataset_name, temp, flux)

temps = [0, 10, -5]
fluxs = [0.5, 1, 2]
for temp in temps:
    for flux in fluxs:
        temp_str = str(temp)
        flux_str = str(flux)
        if flux == 0.5: flux_str="0_5"
        dataset_name = "lukas_T"+temp_str+"_flux"+flux_str+"_pksi"
        ### real dataset
        if not legacy:
            dataset = SimulationDataset("/home/pillerls/heat-plume/datasets/prepared/dataset_test_vary_T_vary_inflow/"+dataset_name)
            create_example_plots(dataset, dataset_name+"2", temp, flux)

# temp_str = str(0)
# flux_str = str(22)
# if flux == 0.5: flux_str="0_5"
# dataset_name = "dataset_test2_pksi"
# ### real dataset
# if not legacy:
#     dataset = SimulationDataset("/home/pillerls/heat-plume/datasets/prepared/dataset_test_vary_T_vary_inflow/"+dataset_name)
#     create_example_plots(dataset, dataset_name, temp, flux)
        