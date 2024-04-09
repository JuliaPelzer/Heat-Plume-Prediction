import torch
import torch.nn as nn
from torch.nn import MSELoss
from physics.equations_of_state import eos_water_saturation_pressure_IFC67, eos_water_viscosity_1, eos_water_enthalphy, eos_water_density_IFC67, thermal_conductivity


def create_loss_fn(dataloaders: dict = None, settings = None):
    if settings.loss == "data":
        loss_class = DataLoss(settings.device)
    elif settings.loss == "physical":
        loss_class = PhysicalLossV2(settings.device)
    elif settings.loss == "mixed":
        loss_class = MixedLoss(settings.device)
    else:
        raise ValueError(f"loss_fn_str: {settings.loss} not implemented")
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

class PhysicalLossV1(BaseLoss): # version 1: correct residual
    def __init__(self, device="cuda:2"):
        super().__init__(device)
        self.weight = [1.0, 1.0, 1.0]

    def __call__(self, input, output, target, dataloader): # permeability index 1 of input
        raise NotImplementedError
    
    def get_continuity_error(self, temperature, pressure, permeability, cell_width):
        density, molar_density = eos_water_density_IFC67(temperature, pressure)
        saturation_pressure = eos_water_saturation_pressure_IFC67(temperature)
        viscosity = eos_water_viscosity_1(temperature, pressure, saturation_pressure)

        alpha_pressure = -1.0 * molar_density * permeability / viscosity

        return self.complex_laplace(alpha_pressure, pressure, cell_width)
    
    def get_darcy_errors(self, temperature, pressure, q_x, q_y, permeability, cell_width):
        dpdx = self.central_differences_x(pressure, cell_width)
        dpdy = self.central_differences_y(pressure, cell_width)
        
        saturation_pressure = eos_water_saturation_pressure_IFC67(temperature)
        viscosity = eos_water_viscosity_1(temperature, pressure, saturation_pressure)

        darcy_x_error = q_x[..., 1:-1, 1:-1] + (permeability[..., 1:-1, 1:-1] / viscosity[..., 1:-1, 1:-1]) * dpdx
        darcy_y_error = q_y[..., 1:-1, 1:-1] + (permeability[..., 1:-1, 1:-1] / viscosity[..., 1:-1, 1:-1]) * dpdy
        return darcy_x_error, darcy_y_error

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

class PhysicalLossV2(BaseLoss): # version 2: simplified residual
    def __init__(self, device="cuda:2"):
        super().__init__(device)
        self.MSE = MSELoss()
        self.weight = [1.0, 1.0]
        self.mask = torch.tensor([[(i == 0) for j in range(64)] for i in range(64)])
        self.downsample = lambda x: nn.functional.interpolate(x, scale_factor=0.5, mode="bicubic")#nn.AvgPool2d(kernel_size=2, stride=2)
        self.weights = torch.tensor([[.25, .25],
                                [.25, .25]], device=self.device) * 8.0
        self.weights = self.weights.view(1, 1, 2, 2)

    def __call__(self, input, output, target, dataloader): # permeability index 1 of input, temperature index 1 and pressure index 0 of label
        dataset = dataloader.dataset.dataset
        norm = dataset.norm
        #output[self.mask.repeat(output.size(0), 2, 1, 1)] = target[self.mask.repeat(output.size(0), 2, 1, 1)]
        input_swap = input.detach().clone().swapaxes(0, 1)
        output_swap = output.clone().swapaxes(0, 1)
        inputs = norm.reverse(input_swap, "Inputs")
        repeats = output.shape[2] // inputs.shape[2]
        permeability = inputs[2,:,:,:].repeat(1, repeats, 1).unsqueeze(1)
        gradient = inputs[1,:,:,:].repeat(1, repeats, 1).unsqueeze(1)
        outputs = norm.reverse(output_swap, "Labels")
        temperature = outputs[0,:,:,:].unsqueeze(1)
        pressure = torch.tensor(848673.4375)
        # boundary_error = torch.mean(torch.pow(output[:, 0][self.mask.repeat(temperature.size(0), 1, 1)] - target[:, 0][self.mask.repeat(temperature.size(0), 1, 1)], 2))
        # boundary_error += torch.mean(torch.pow(output[:, 1][self.mask.repeat(temperature.size(0), 1, 1)] - target[:, 1][self.mask.repeat(temperature.size(0), 1, 1)], 2))
        # temperature_error = torch.mean(torch.pow(torch.minimum(temperature-10.6, torch.zeros_like(temperature)), 2)) + torch.mean(torch.pow(torch.maximum(temperature - 15.6, torch.zeros_like(temperature)), 2))

        cell_width = dataset.info["CellsSize"][0]

        # continuity_error = self.get_continuity_error(temperature[:, :, 62:], gradient[:, :, 62:], pressure, permeability[:, :, 62:], cell_width)
        # continuity_error[:, :, 0:3] = continuity_error[:, :, 0:3] * 10.0
        
        energy_error = self.get_energy_error(temperature[:, :, 62:], gradient[:, :, 62:], pressure, permeability[:, :, 62:], cell_width)
        # energy_error[:, :, 0:3] = energy_error[:, :, 0:3] * 2.0

        # TODO is this useful or unnecessary? Initial experiment says useful
        # Idea is that averaging and scaling cancels out in 'random' are but not in end of plume or similar ones

        energy_error_downsampled = nn.functional.conv2d(energy_error, self.weights)

        # physics_loss_orig = self.temporally_weighted_loss(energy_error)
        physics_loss_orig = torch.mean(torch.pow(energy_error_downsampled, 2))
        # physics_loss_orig = torch.mean(torch.pow(energy_error, 2)) #+ torch.mean(torch.pow(continuity_error, 2))

        #physics_loss_orig = self.weight[0] * torch.mean(torch.pow(continuity_error, 2)) + self.weight[1] * torch.mean(torch.pow(energy_error, 2))
        #physics_loss_orig = self.weight[0] * torch.mean(torch.abs(continuity_error)) + self.weight[1] * torch.mean(torch.abs(energy_error)) # 
        # continuity_error_dx = self.central_differences_x(continuity_error, cell_width)
        # continuity_error_dy = self.central_differences_y(continuity_error, cell_width)
        # energy_error_dx = self.central_differences_x(energy_error, cell_width)
        # energy_error_dy = self.central_differences_y(energy_error, cell_width)


        # multigrid losses
        # temperature_grid_2 = self.downsample(temperature)
        # pressure_grid_2 = self.downsample(pressure)
        # permeability_grid_2 = self.downsample(permeability)
        # cell_width_grid_2 = cell_width * 2
        # physics_loss_grid_2 = self.get_physical_loss(temperature_grid_2, pressure_grid_2, permeability_grid_2, cell_width_grid_2)

        # temperature_grid_4 = self.downsample(temperature_grid_2)
        # pressure_grid_4 = self.downsample(pressure_grid_2)
        # permeability_grid_4 = self.downsample(permeability_grid_2)
        # cell_width_grid_4 = cell_width_grid_2 * 2
        # physics_loss_grid_4 = self.get_physical_loss(temperature_grid_4, pressure_grid_4, permeability_grid_4, cell_width_grid_4)

        return physics_loss_orig #+ 1e4 *  boundary_error +  (torch.max(torch.abs(continuity_error_dx)) + torch.max(torch.abs(continuity_error_dy)) + torch.max(torch.abs(energy_error_dx)) + torch.max(torch.abs(energy_error_dy)))

    def temporally_weighted_loss(self, residual):
        epsilon = -100.0
        residual = torch.mean(torch.pow(residual, 2), (0, 1, 3))
        N = residual.shape[0]
        weights = torch.zeros(N, device=residual.device)
        partial_sum = torch.tensor(0.0)
        for i in range(N):
            partial_sum = partial_sum + residual[i]
            weights[i] = torch.exp(epsilon * partial_sum)
        return torch.mean(weights.detach() * residual)

    def get_physical_loss(self, temperature, pressure, permeability, cell_width):
        continuity_error = self.get_continuity_error(temperature, pressure, permeability, cell_width)
        
        energy_error = self.get_energy_error(temperature, pressure, permeability, cell_width)

        #physics_loss = self.weight[0] * torch.mean(torch.pow(continuity_error, 2)) + self.weight[1] * torch.mean(torch.pow(energy_error, 2))
        return self.weight[0] * torch.mean(torch.abs(continuity_error)) + self.weight[1] * torch.mean(torch.abs(energy_error))

    def get_darcy(self, temperature, pressure, permeability, cell_width):
        dpdx = self.central_differences_x(pressure, cell_width)
        dpdy = self.central_differences_y(pressure, cell_width)
        
        saturation_pressure = eos_water_saturation_pressure_IFC67(temperature)
        viscosity = eos_water_viscosity_1(temperature, pressure, saturation_pressure)

        q_x = -1.0 * (permeability[..., 1:-1, 1:-1] / viscosity[..., 1:-1, 1:-1]) * dpdx
        q_y = -1.0 * (permeability[..., 1:-1, 1:-1] / viscosity[..., 1:-1, 1:-1]) * dpdy
        return q_x, q_y


    def get_continuity_error(self, temperature, gradient, pressure, permeability, cell_width):
        density, molar_density = eos_water_density_IFC67(temperature, pressure)

        q_x = -1.0 * gradient * permeability *1000*1000*9.81

        cont_error_pressure = self.central_differences_x(molar_density * q_x, cell_width)

        return cont_error_pressure
    

    def get_energy_error(self, temperature, gradient, pressure, permeability, cell_width):
        # thermal conductivity is constant
        thermal_conductivity = 1.0
    
        density, molar_density = eos_water_density_IFC67(temperature, pressure)
        enthalpy = eos_water_enthalphy(temperature, pressure)

        alpha_pressure = molar_density * enthalpy

        q_x = -1.0 * gradient * permeability *1000*1000*9.81

        energy_error_pressure = self.central_differences_x(alpha_pressure * q_x, cell_width)
        energy_error_temperature = -1.0 * thermal_conductivity * self.laplace(temperature, cell_width)
        
        # print("enthalpy:", enthalpy[..., 22, 6])
        # print("energy_error_pressure:", energy_error_pressure[..., 22, 6])
        # print("energy_error_temperature:", energy_error_temperature[..., 22, 6])
        # print("energy_error:", energy_error_pressure[..., 22, 6] + energy_error_temperature[..., 22, 6])

        return energy_error_pressure + energy_error_temperature


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