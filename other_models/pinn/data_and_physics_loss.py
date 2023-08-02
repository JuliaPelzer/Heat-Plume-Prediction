import numpy as np
import torch.nn as nn
from torch import Tensor, autograd
import torch
from typing import List
from other_models.pinn.equations_of_state import eos_water_saturation_pressure_IFC67, eos_water_viscosity_1, eos_water_enthalpy, eos_water_density_IFC67, thermal_conductivity

# main balance functions
def darcy(p:Tensor, t:Tensor, k:Tensor, params:dict):
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

def continuity(p:Tensor, t:Tensor, q:List[Tensor], pos_inflow:np.ndarray, params:dict):
    density, molar_density = eos_water_density_IFC67(t, p)

    rhs = pos_inflow * molar_density * params["inflow"] /(params["cell_length"]**3)/2
    dudx_cont = torch.gradient(molar_density * q[0], spacing=params["cell_length"])[1]
    dvdy_cont = torch.gradient(molar_density * q[1], spacing=params["cell_length"])[0]
    continuity = dudx_cont + dvdy_cont -rhs
    return continuity

def energy(p:Tensor, t:Tensor, q:List[Tensor], pos_inflow:np.ndarray, params:dict):
    enthalpy = eos_water_enthalpy(t, p)
    thermal_conductivity = params["thermal_conductivity"]
    density, molar_density = eos_water_density_IFC67(t, p)

    rhs = pos_inflow * density * params["inflow"] * params["heat capacity"] * params["delta T"] / (params["cell_length"]**3)
    dTdy, dTdx = torch.gradient(t, spacing=params["cell_length"])
    energy_inner_u = molar_density * q[0] * enthalpy - thermal_conductivity * dTdx
    energy_inner_v = molar_density * q[1] * enthalpy - thermal_conductivity * dTdy

    energy_u = torch.gradient(energy_inner_u, spacing=params["cell_length"])[1]
    energy_v = torch.gradient(energy_inner_v, spacing=params["cell_length"])[0]
    energy = energy_u + energy_v - rhs
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

def van_genuchten_pc(alpha:float, m:float, capillary_pressure:Tensor):
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
        weight_init = weight_physics * 0.25
        self.weights_physics = Tensor([1,1,1,1]) * weight_init
        self.params_physics = {"cell_length" : 5,
        	"genuchten_m" : 0.5,
            "saturation_liquid_residual" : 0.1,
            "thermal_conductivity" : 1,
            "inflow" : 0.00024, #m^3/s
            "heat capacity" : 4200, #J/(kg*K)
            "delta T": 5, #K
            }
        self.loss_func : nn.modules.loss._Loss = nn.MSELoss() 
        self.norm = norm

    def forward(self, prediction: Tensor, input: Tensor, target: Tensor) -> Tensor:
        # TODO whatabout BC, IC
        input = input.clone()
        target = target.clone()
        prediction_local = prediction.detach().clone()

        q_u_pred = Tensor(target[:,0,:,:].shape)
        q_v_pred = Tensor(target[:,0,:,:].shape)
        conti_pred = Tensor(target[:,0,:,:].shape)
        energy_pred = Tensor(target[:,0,:,:].shape)
        q_u_target = Tensor(target[:,0,:,:].shape)
        q_v_target = Tensor(target[:,0,:,:].shape)
        conti_target = Tensor(target[:,0,:,:].shape)
        energy_target = Tensor(target[:,0,:,:].shape)

        for datapoint_idx in range(input.shape[0]):
            input_dp = self.norm.reverse(input[datapoint_idx], "Inputs")
            # input_dp = input[datapoint_idx]
            target_dp = self.norm.reverse(target[datapoint_idx], "Labels")
            prediction_dp = self.norm.reverse(prediction_local[datapoint_idx], "Labels")

            input_dp = autograd.Variable(input_dp, requires_grad=True)
            target_dp = autograd.Variable(target_dp, requires_grad=True)
            prediction_dp = autograd.Variable(prediction_dp, requires_grad=True)

            p_orig = input_dp[0]  # TODO make this more general (get index from info.yaml)
            k_orig = input_dp[1]
            pos_orig_datapoint = input_dp[3]
            t_target = target_dp[0]
            t_pred = prediction_dp[0]

            q_u_pred_datapoint, q_v_pred_datapoint = darcy(p_orig, t_pred, k_orig, self.params_physics)
            conti_pred_datapoint = continuity(p_orig, t_pred, [q_u_pred_datapoint, q_v_pred_datapoint], pos_orig_datapoint, self.params_physics)
            energy_pred_datapoint = energy(p_orig, t_pred, [q_u_pred_datapoint, q_v_pred_datapoint], pos_orig_datapoint, self.params_physics)

            q_u_target_datapoint, q_v_target_datapoint = darcy(p_orig, t_target, k_orig, self.params_physics)
            conti_target_datapoint = continuity(p_orig, t_target, [q_u_target_datapoint, q_v_target_datapoint], pos_orig_datapoint, self.params_physics)
            energy_target_datapoint = energy(p_orig, t_target, [q_u_target_datapoint, q_v_target_datapoint], pos_orig_datapoint, self.params_physics)

            q_u_pred[datapoint_idx] = q_u_pred_datapoint
            q_v_pred[datapoint_idx] = q_v_pred_datapoint
            conti_pred[datapoint_idx] = conti_pred_datapoint
            energy_pred[datapoint_idx] = energy_pred_datapoint
            q_u_target[datapoint_idx] = q_u_target_datapoint
            q_v_target[datapoint_idx] = q_v_target_datapoint
            conti_target[datapoint_idx] = conti_target_datapoint
            energy_target[datapoint_idx] = energy_target_datapoint

        data_loss = 0 #(1-sum(self.weights_physics)) * self.loss_func(prediction, target)
        physics_loss_qu = self.weights_physics[0] * self.loss_func(q_u_pred, q_u_target)
        physics_loss_qv = self.weights_physics[1] * self.loss_func(q_v_pred, q_v_target)
        physics_loss_conti = self.weights_physics[2] * self.loss_func(conti_pred, conti_target)
        physics_loss_energy = self.weights_physics[3] * self.loss_func(energy_pred, energy_target)

        # TODO + physics_loss_boundary + physics_loss_initial
        # TODO include source terms ?
        return data_loss + physics_loss_qu + physics_loss_qv + physics_loss_conti + physics_loss_energy