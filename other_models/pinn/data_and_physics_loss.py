import numpy as np
import torch.nn as nn
from torch import Tensor
import torch
from typing import List
from other_models.pinn.equations_of_state import eos_water_saturation_pressure_IFC67, eos_water_viscosity_1, eos_water_enthalpy, eos_water_density_IFC67, thermal_conductivity

# main balance functions
def darcy(p:np.ndarray, t:np.ndarray, k:np.ndarray, params:dict):
    genuchten_m = params["genuchten_m"]
    saturation_liquid_residual = params["saturation_liquid_residual"]
    saturation = 1.0 * torch.ones_like(p)

    saturation_pressure = eos_water_saturation_pressure_IFC67(t)
    viscosity = eos_water_viscosity_1(t, p, saturation_pressure)
    saturation_effective = van_genuchten(saturation_max=1, saturation_residual=saturation_liquid_residual, saturation=saturation)
    k_rel = mualem_vg_liq(genuchten_m, saturation_effective)
    
    dpdy, dpdx = torch.gradient(p, spacing=params["cell_length"])
    q_u = (- k * k_rel / viscosity) * dpdx
    q_v = (- k * k_rel/ viscosity) * dpdy

    return q_u, q_v

def continuity(p:np.ndarray, t:np.ndarray, q:List[np.ndarray], params:dict):
    density, molar_density = eos_water_density_IFC67(t, p)
    dudx_cont = torch.gradient(molar_density * q[0], spacing=params["cell_length"])[1]
    dvdy_cont = torch.gradient(molar_density * q[1], spacing=params["cell_length"])[0]
    continuity = dudx_cont + dvdy_cont
    return continuity

def energy(p:np.ndarray, t:np.ndarray, q:List[np.ndarray], params:dict):
    enthalpy = eos_water_enthalpy(t, p)
    thermal_conductivity = params["thermal_conductivity"]
    density, molar_density = eos_water_density_IFC67(t, p)

    dTdy, dTdx = torch.gradient(t, spacing=params["cell_length"])
    energy_inner_u = molar_density * q[0] * enthalpy - thermal_conductivity * dTdx
    energy_inner_v = molar_density * q[1] * enthalpy - thermal_conductivity * dTdy

    energy_u = torch.gradient(energy_inner_u, spacing=params["cell_length"])[1]
    energy_v = torch.gradient(energy_inner_v, spacing=params["cell_length"])[0]
    energy = energy_u + energy_v
    return energy

# helper functions
def mualem_vg_liq(m, saturation_effective):
    """ 
    PERMEABILITY_FUNCTION MUALEM_VG_LIQ, from https://documentation.pflotran.org/theory_guide/constitutive_relations.html

    Parameters:
      m: van Genuchten m parameter, as in (m = 1-1/n) or (m = 1 - 2/n) [-].
      saturation_effective: effective saturation [-]. 
      If not constant: use formula below and LIQUID_RESIDUAL_SATURATION 0.1d0 LIQUID_RESIDUAL_SATURATION <float>    Residual saturation for liquid phase [-].
    Returns:
      relative permeability for liquid phase [-].
    """
    # saturation_effective = (saturation-residual_saturation)/(s_max-residual_saturation)
    outer_pow = torch.pow(1-torch.pow(saturation_effective, 1/m), m)
    return  torch.sqrt(saturation_effective)*torch.pow(1-outer_pow, 2)

def van_genuchten(saturation, saturation_residual, saturation_max):
    """ 
    PERMEABILITY_FUNCTION VAN_GENUCHTEN, from https://documentation.pflotran.org/theory_guide/constitutive_relations.html

    Parameters
      saturation: saturation [-].
      saturation_residual: residual saturation [-].
      saturation_max: maximum saturation [-].

    Returns
      effective saturation [-].
    """
    return (saturation-saturation_residual)/(saturation_max-saturation_residual)

def van_genuchten_pc(alpha:float, m:float, capillary_pressure:np.ndarray):
    """
    PERMEABILITY_FUNCTION VAN_GENUCHTEN_PC, from https://documentation.pflotran.org/theory_guide/constitutive_relations.html

    Parameters
      alpha: van Genuchten alpha parameter [1/Pa].
      m: van Genuchten m parameter, as in (m = 1-1/n) or (m = 1 - 2/n) [-].
      capillary_pressure: capillary pressure [Pa].

    Returns
      effective saturation [-].
    """
    n = 1.0/ (1.0 - m)
    return torch.pow(1.0 + torch.pow(alpha * capillary_pressure, n), -m)
    
# combined loss function
class DataAndPhysicsLoss(nn.modules.loss._Loss):
    def __init__(self, norm, weight_physics: float = 0.01) -> None:
        super(nn.modules.loss._Loss, self).__init__()
        self.weight_physics = weight_physics
        self.params_physics = {"cell_length" : 5,
            "genuchten_m" : 0.5,
            "saturation_liquid_residual" : 0.1,
            "thermal_conductivity" : 1,
            }
        self.loss_func : nn.modules.loss._Loss = nn.MSELoss() 
        self.norm = norm

    def forward(self, prediction: Tensor, input: Tensor, target: Tensor) -> Tensor:
        # TODO whatabout BC, IC
        # make tensor q_u_pred in size of target
        q_u_pred = Tensor(target[:,0,:,:].shape)
        q_v_pred = Tensor(target[:,0,:,:].shape)
        conti_pred = Tensor(target[:,0,:,:].shape)
        energy_pred = Tensor(target[:,0,:,:].shape)
        q_u_target = Tensor(target[:,0,:,:].shape)
        q_v_target = Tensor(target[:,0,:,:].shape)
        conti_target = Tensor(target[:,0,:,:].shape)
        energy_target = Tensor(target[:,0,:,:].shape)

        for datapoint_idx in range(input.shape[0]):
            input[datapoint_idx] = self.norm.reverse(input[datapoint_idx], "Inputs")
            target[datapoint_idx] = self.norm.reverse(target[datapoint_idx], "Labels")
            prediction[datapoint_idx] = self.norm.reverse(prediction[datapoint_idx], "Labels")

            p_orig = input[datapoint_idx, 0, :, :]  # TODO make this more general (get index from info.yaml)
            k_orig = input[datapoint_idx, 1, :, :]
            t_target = target[datapoint_idx, 0, :, :]
            t_pred = prediction[datapoint_idx, 0, :, :]

            q_u_pred_datapoint, q_v_pred_datapoint = darcy(p_orig, t_pred, k_orig, self.params_physics)
            conti_pred_datapoint = continuity(p_orig, t_pred, [q_u_pred_datapoint, q_v_pred_datapoint], self.params_physics)
            energy_pred_datapoint = energy(p_orig, t_pred, [q_u_pred_datapoint, q_v_pred_datapoint], self.params_physics)

            q_u_target_datapoint, q_v_target_datapoint = darcy(p_orig, t_target, k_orig, self.params_physics)
            conti_target_datapoint = continuity(p_orig, t_target, [q_u_target_datapoint, q_v_target_datapoint], self.params_physics)
            energy_target_datapoint = energy(p_orig, t_target, [q_u_target_datapoint, q_v_target_datapoint], self.params_physics)
            if datapoint_idx == 0:
                import matplotlib.pyplot as plt
                fig = plt.figure()
                ax1 = fig.add_subplot(211)
                ax2 = fig.add_subplot(212)
                ax1.imshow(q_v_pred_datapoint.cpu().detach().numpy().T)
                ax1.set_title("q_v_pred")
                ax2.imshow(q_v_target_datapoint.cpu().detach().numpy().T)
                ax2.set_title("q_v_target")
                plt.show()
            # exit()
            q_u_pred[datapoint_idx] = q_u_pred_datapoint
            q_v_pred[datapoint_idx] = q_v_pred_datapoint
            conti_pred[datapoint_idx] = conti_pred_datapoint
            energy_pred[datapoint_idx] = energy_pred_datapoint
            q_u_target[datapoint_idx] = q_u_target_datapoint
            q_v_target[datapoint_idx] = q_v_target_datapoint
            conti_target[datapoint_idx] = conti_target_datapoint
            energy_target[datapoint_idx] = energy_target_datapoint

        return self.loss_func(prediction, target) + self.weight_physics/4 * (self.loss_func(q_u_pred, q_u_target) + self.loss_func(q_v_pred, q_v_target) + self.loss_func(conti_pred, conti_target) + self.loss_func(energy_pred, energy_target))