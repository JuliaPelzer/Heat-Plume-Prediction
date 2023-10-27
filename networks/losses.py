import torch
import torch.nn as nn
from torch.nn import MSELoss
from physics.equations_of_state import eos_water_saturation_pressure_IFC67, eos_water_viscosity_1, eos_water_enthalphy, eos_water_density_IFC67, thermal_conductivity


def create_loss_fn(loss_fn_str: str, dataloaders: dict = None):
    if loss_fn_str == "data":
        loss_class = DataLoss()
    elif loss_fn_str == "physical":
        loss_class = PhysicalLossV2()
    elif loss_fn_str == "mixed":
        loss_class = MixedLoss()
    else:
        raise ValueError(f"loss_fn_str: {loss_fn_str} not implemented")
    return loss_class

class BaseLoss():
    def __init__(self, device: str ="cuda:2"):
        super().__init__()
        self.device = device
    
    def __call__(self, input: torch.tensor, output: torch.tensor, target: torch.tensor, dataloader):
        raise NotImplementedError

class DataLoss(BaseLoss):
    def __init__(self, device="cuda:2"):
        super().__init__(device)
        self.MSE = MSELoss()
    
    def __call__(self, input, output, target, dataloader):
        return self.MSE(output, target)

class PhysicalLoss(BaseLoss): # version 1: with all ground truths
    def __init__(self, device="cuda:2"):
        super().__init__(device)
        self.weight = [1.0, 1.0, 1.0]

    def __call__(self, input, output, target, dataloader): # permeability index 1 of input
        norm = dataloader.dataset.dataset.norm
        input_swap = input.detach().clone().swapaxes(0, 1)
        permeability = norm.reverse(input_swap, "Inputs")[1,:,:,:].unsqueeze(1)
        temperature = norm.reverse(output.clone()[:,0,:,:], "Labels").unsqueeze(1)
        q_x = output[:,1:2,:,:]
        q_y = output[:,2:3,:,:]
        pressure = output[:,3:4,:,:]

        cell_width = 5

        continuity_error = self.get_continuity_error(temperature, pressure, q_x, q_y, cell_width)

        darcy_x_error, darcy_y_error = self.get_darcy_errors(temperature, pressure, q_x, q_y, permeability, cell_width)
        
        energy_error = self.get_energy_error(temperature, pressure, q_x, q_y, cell_width) 


        return self.weight[0] * torch.mean(torch.pow(continuity_error, 2)) + self.weight[1] * torch.mean(torch.pow(darcy_x_error, 2)) + self.weight[1] * torch.mean(torch.pow(darcy_y_error, 2)) + self.weight[2] * torch.mean(torch.pow(energy_error, 2))
    
    def get_continuity_error(self, temperature, pressure, q_x, q_y, cell_width):
        density, molar_density = eos_water_density_IFC67(temperature, pressure)
        return self.central_differences_x(molar_density * q_x, cell_width) + self.central_differences_y(molar_density * q_y, cell_width)  # mistake around pump
    
    def get_darcy_errors(self, temperature, pressure, q_x, q_y, permeability, cell_width):
        dpdx = self.central_differences_x(pressure, cell_width)
        dpdy = self.central_differences_y(pressure, cell_width)
        
        saturation_pressure = eos_water_saturation_pressure_IFC67(temperature)
        viscosity = eos_water_viscosity_1(temperature, pressure, saturation_pressure)

        darcy_x_error = q_x[..., 1:-1, 1:-1] + (permeability[..., 1:-1, 1:-1] / viscosity[..., 1:-1, 1:-1]) * dpdx
        darcy_y_error = q_y[..., 1:-1, 1:-1] + (permeability[..., 1:-1, 1:-1] / viscosity[..., 1:-1, 1:-1]) * dpdy
        return darcy_x_error, darcy_y_error

    def get_energy_error(self, temperature, pressure, q_x, q_y, cell_width):
        thermal_conductivity = 1.0
    
        density, molar_density = eos_water_density_IFC67(temperature, pressure)
        enthalpy = eos_water_enthalphy(temperature, pressure)
        
        energy_error_darcy = self.central_differences_x(molar_density * q_x * enthalpy, cell_width) + self.central_differences_y(molar_density * q_y * enthalpy, cell_width)

        energy_error_temperature = self.laplace(thermal_conductivity * temperature, cell_width)
        
        return energy_error_darcy - energy_error_temperature # mistake around pump



    def central_differences_x(self, values: torch.tensor, h = 1.0):
        weights = torch.tensor([[0., -1., 0.],
                        [0., 0., 0.],
                        [0., 1., 0.]], device=self.device)
        weights = weights.view(1, 1, 3, 3)

        ans = nn.functional.conv2d(values, weights)
        return ans / (2 * h)
    
    def central_differences_y(self, values: torch.tensor, h = 1.0):
        weights = torch.tensor([[0., 0., 0.],
                        [-1., 0., 1.],
                        [0., 0., 0.]], device=self.device)
        weights = weights.view(1, 1, 3, 3)

        ans = nn.functional.conv2d(values, weights)
        return ans / (2 * h)
    
    def laplace(self, values: torch.tensor, h = 1.0):
        weights = torch.tensor([[0., 1., 0.],
                        [1., -4., 1.],
                        [0., 1., 0.]], device=self.device)
        weights = weights.view(1, 1, 3, 3)

        ans = nn.functional.conv2d(values, weights)
        return ans / (h**2)

class PhysicalLossV2(BaseLoss): # version 2: without darcy velocity and different stencil with constancy assumption
    def __init__(self, device="cuda:2"):
        super().__init__(device)
        self.MSE = MSELoss()
        self.weight = [1.0, 0.001]

    def __call__(self, input, output, target, dataloader): # permeability index 1 of input, temperature index 1 and pressure index 0 of label
        dataset = dataloader.dataset.dataset
        norm = dataset.norm
        input_swap = input.detach().clone().swapaxes(0, 1)
        output_swap = output.clone().swapaxes(0, 1)
        permeability = norm.reverse(input_swap, "Inputs")[1,:,:,:].unsqueeze(1)
        prediction = norm.reverse(output_swap, "Labels")
        temperature = prediction[1].squeeze().unsqueeze(1)
        pressure = prediction[0].squeeze().unsqueeze(1)

        cell_width = dataset.info["CellsSize"][0]

        continuity_error = self.get_continuity_error(temperature, pressure, permeability, cell_width)
        
        energy_error = self.get_energy_error(temperature, pressure, permeability, cell_width)

        hp_pos = dataset.info["PositionLastHP"][:2]
        # continuity_error[:, :, hp_pos[1]-1, hp_pos[0]-1] = 0.0
        # energy_error[:, :, hp_pos[1]-1, hp_pos[0]-1] = 0.0
        temperature_hp = torch.tensor(15.6)
        pressure_hp = pressure[:, :, hp_pos[1], hp_pos[0]]

        inflow = torch.tensor(0.00024)# convert to m^3/s
        cell_volume = torch.tensor(cell_width) * cell_width * cell_width #m^3
        _, molar_density = eos_water_density_IFC67(temperature_hp, pressure_hp) # kmol/m^3
        source_continuity = inflow / cell_volume * molar_density

        source_energy = source_continuity * (eos_water_enthalphy(temperature_hp, pressure_hp) - pressure_hp / molar_density / 1000)  #TODO fix this

        continuity_error[:, :, hp_pos[1]-1, hp_pos[0]-1] -= source_continuity
        energy_error[:, :, hp_pos[1]-1, hp_pos[0]-1] -= source_energy

        physics_loss = self.weight[0] * torch.mean(torch.pow(continuity_error, 4)) + self.weight[1] * torch.mean(torch.pow(energy_error, 4))
        # continuity_error_dx = self.central_differences_x(continuity_error, cell_width)
        # continuity_error_dy = self.central_differences_y(continuity_error, cell_width)
        # energy_error_dx = self.central_differences_x(energy_error, cell_width)
        # energy_error_dy = self.central_differences_y(energy_error, cell_width)
        return physics_loss #+ torch.max(continuity_error_dx) + torch.max(continuity_error_dy) + torch.max(energy_error_dx) + torch.max(energy_error_dy)
    

    def get_darcy(self, temperature, pressure, permeability, cell_width):
        dpdx = self.central_differences_x(pressure, cell_width)
        dpdy = self.central_differences_y(pressure, cell_width)
        
        saturation_pressure = eos_water_saturation_pressure_IFC67(temperature)
        viscosity = eos_water_viscosity_1(temperature, pressure, saturation_pressure)

        q_x = -1.0 * (permeability[..., 1:-1, 1:-1] / viscosity[..., 1:-1, 1:-1]) * dpdx
        q_y = -1.0 * (permeability[..., 1:-1, 1:-1] / viscosity[..., 1:-1, 1:-1]) * dpdy
        return q_x, q_y


    def get_continuity_error(self, temperature, pressure, permeability, cell_width):
        density, molar_density = eos_water_density_IFC67(temperature, pressure)
        saturation_pressure = eos_water_saturation_pressure_IFC67(temperature)
        viscosity = eos_water_viscosity_1(temperature, pressure, saturation_pressure)

        alpha_pressure = -1.0 * molar_density * permeability / viscosity

        return self.complex_laplace(alpha_pressure, pressure, cell_width)  # mistake around pump
    

    def get_energy_error(self, temperature, pressure, permeability, cell_width):
        # thermal conductivity is constant
        thermal_conductivity = 1.0
    
        density, molar_density = eos_water_density_IFC67(temperature, pressure)
        enthalpy = eos_water_enthalphy(temperature, pressure)
        saturation_pressure = eos_water_saturation_pressure_IFC67(temperature)
        viscosity = eos_water_viscosity_1(temperature, pressure, saturation_pressure)

        alpha_pressure = -1.0 * molar_density * permeability / viscosity * enthalpy

        energy_error_pressure = self.complex_laplace(alpha_pressure, pressure, cell_width)
        energy_error_temperature = -1.0 * thermal_conductivity * self.laplace(temperature, cell_width)
        
        # print("enthalpy:", enthalpy[..., 22, 6])
        # print("energy_error_pressure:", energy_error_pressure[..., 22, 6])
        # print("energy_error_temperature:", energy_error_temperature[..., 22, 6])
        # print("energy_error:", energy_error_pressure[..., 22, 6] + energy_error_temperature[..., 22, 6])

        return energy_error_pressure + energy_error_temperature # mistake around pump


    def central_differences_x(self, values: torch.tensor, h = 1.0):
        weights = torch.tensor([[0., -1., 0.],
                        [0., 0., 0.],
                        [0., 1., 0.]], device=self.device)
        weights = weights.view(1, 1, 3, 3)

        ans = nn.functional.conv2d(values, weights)
        return ans / (2 * h)
    
    def central_differences_y(self, values: torch.tensor, h = 1.0):
        weights = torch.tensor([[0., 0., 0.],
                        [-1., 0., 1.],
                        [0., 0., 0.]], device=self.device)
        weights = weights.view(1, 1, 3, 3)

        ans = nn.functional.conv2d(values, weights)
        return ans / (2 * h)
    
    def laplace(self, values: torch.tensor, h = 1.0):
        weights = torch.tensor([[0., 1., 0.],
                        [1., -4., 1.],
                        [0., 1., 0.]], device=self.device)
        weights = weights.view(1, 1, 3, 3)

        ans = nn.functional.conv2d(values, weights)
        return ans / (h**2)
    
    def complex_laplace(self, alpha: torch.tensor, values: torch.tensor, h = 1.0):
        assert alpha.shape == values.shape
        N, M = alpha.shape[-2:]
        
        result = torch.zeros(alpha.shape[:-2] + (N-2, M-2), device=alpha.device)
        # for i in range(1, N-1):
        #     for j in range(1, M-1):
        #         val_left  = (alpha[..., i, j] + alpha[..., i-1, j]) / 2 * values[..., i-1, j]
        #         val_right = (alpha[..., i, j] + alpha[..., i+1, j]) / 2 * values[..., i+1, j]
        #         val_top   = (alpha[..., i, j] + alpha[..., i, j-1]) / 2 * values[..., i, j-1]
        #         val_bott  = (alpha[..., i, j] + alpha[..., i, j+1]) / 2 * values[..., i, j+1]
        #         val_cent  = (alpha[..., i, j] * 4 + alpha[..., i-1, j] + alpha[..., i+1, j] + alpha[..., i, j-1] + alpha[..., i, j+1]) / 2 * values[..., i, j]
        #         result[..., i-1, j-1] = (val_left + val_right + val_top + val_bott - val_cent) / h**2

        val_left  = (alpha[..., 1:N-1, 1:M-1] + alpha[..., 0:N-2, 1:M-1]) * values[..., 0:N-2, 1:M-1]
        val_right = (alpha[..., 1:N-1, 1:M-1] + alpha[..., 2:N, 1:M-1])   * values[..., 2:N, 1:M-1]
        val_top   = (alpha[..., 1:N-1, 1:M-1] + alpha[..., 1:N-1, 0:M-2]) * values[..., 1:N-1, 0:M-2]
        val_bott  = (alpha[..., 1:N-1, 1:M-1] + alpha[..., 1:N-1, 2:M])   * values[..., 1:N-1, 2:M]
        val_cent  = (alpha[..., 1:N-1, 1:M-1] * 4 + alpha[..., 0:N-2, 1:M-1] + alpha[..., 2:N, 1:M-1] + alpha[..., 1:N-1, 0:M-2] + alpha[..., 1:N-1, 2:M]) * values[..., 1:N-1, 1:M-1]
        result = 0.5 * (val_left + val_right + val_top + val_bott - val_cent) / h**2

        return result
    

class MixedLoss(BaseLoss):
    def __init__(self, device="cuda:2"):
        super().__init__(device)
        self.data_loss = DataLoss(self.device)
        self.physical_loss = PhysicalLossV2(self.device)


    def __call__(self, input, output, target, dataloader):
        return self.data_loss(input, output, target, dataloader) + self.physical_loss(input, output, target, dataloader) / 1e4 # fix this