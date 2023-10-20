import torch
import torch.nn as nn
from torch.nn import MSELoss
from physics.equations_of_state import eos_water_saturation_pressure_IFC67, eos_water_viscosity_1, eos_water_enthalphy, eos_water_density_IFC67, thermal_conductivity


class MSELossExcludeNotChangedTemp(nn.MSELoss):
    def __init__(self, size_average=None, reduction: str = 'mean', ignore_temp: float = 0.0) -> None:
        super(nn.MSELoss, self).__init__(size_average, reduction)
        self.ignore_temp = ignore_temp

    def forward(self, inputs: torch.tensor, targets: torch.tensor):
        assert inputs.shape == targets.shape, f"inputs.shape: {inputs.shape}, targets.shape: {targets.shape}"
        # check where temperature is not the ignore temperature in inputs
        mask_inputs = inputs != self.ignore_temp
        # check where temperature is not the ignore temperature in targets
        mask_targets = targets != self.ignore_temp
        mask = torch.logical_or(mask_inputs, mask_targets)
        out = torch.mean((inputs[mask]-targets[mask])**2)
        return out


class MSELossExcludeYBoundary(nn.MSELoss):
    """ loss specifically excludes a little of the y-boundary because there are some artifacts there (from the simulation)"""

    def __init__(self, **kwargs) -> None:
        super(nn.MSELoss, self).__init__(**kwargs)

    def forward(self, inputs: torch.tensor, targets: torch.tensor):
        # expects input in shape (batch, channels, height, width)
        assert len(
            inputs.shape) == 4, f"inputs.shape: {inputs.shape}, expects (batch, channels, height, width)"
        out = torch.mean((inputs[:, :, 1:-2, :]-targets[:, :, 1:-2, :])**2)
        return out


class InfinityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.tensor, targets: torch.tensor):
        out = torch.max(torch.abs(inputs-targets))
        return out


def normalize(x, mean, std):
    return (x - mean) / std


def create_loss_fn(loss_fn_str: str, dataloaders: dict = None):
    if loss_fn_str == "ExcludeNotChangedTemp":
        ignore_temp = 10.6
        label_stats = dataloaders["train"].dataset.dataset.info["Labels"]
        temp_mean, temp_std = (
            label_stats["Temperature [C]"]["mean"],
            label_stats["Temperature [C]"]["std"],
        )
        normalized_ignore_temp = normalize(ignore_temp, temp_mean, temp_std)
        print(temp_std, temp_mean, normalized_ignore_temp)
        loss_class = MSELossExcludeNotChangedTemp(
            ignore_temp=normalized_ignore_temp)
    elif loss_fn_str == "data":
        loss_class = DataLoss()
    elif loss_fn_str == "ExcludeYBoundary":
        loss_class = MSELossExcludeYBoundary()
    elif loss_fn_str == "Infinity":
        loss_class = InfinityLoss()
    elif loss_fn_str == "physical":
        loss_class = PhysicalLoss()
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
        return self.MSE(output[:,0:1,:,:], target[:,0:1,:,:])

class PhysicalLoss(BaseLoss):
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


        return self.weight[0] * torch.sum(torch.pow(continuity_error, 2)) + self.weight[1] * torch.sum(torch.pow(darcy_x_error, 2)) + self.weight[1] * torch.sum(torch.pow(darcy_y_error, 2)) + self.weight[2] * torch.sum(torch.pow(energy_error, 2))
    
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

        

class MixedLoss(BaseLoss):
    def __init__(self, device="cuda:2"):
        super().__init__(device)
        self.data_loss = DataLoss(self.device)
        self.physical_loss = PhysicalLoss(self.device)


    def __call__(self, input, output, target, dataloader):
        return self.data_loss(input, output, target, dataloader) + self.physical_loss(input, output, target, dataloader) / 1e20