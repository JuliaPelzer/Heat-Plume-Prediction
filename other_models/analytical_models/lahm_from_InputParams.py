from dataclasses import dataclass
import torch
import pathlib
import yaml

import analytical_model_lahm as lahm
import analytical_model_pahm as pahm
import analytical_model_rhm as rhm
import analytical_steady_state_model_willibald as willibald
import numpy as np
import utils_and_visu as utils



@dataclass
class Domain:
    cell_size = 1
    x_lin = np.linspace(0, 1280, int(1280/cell_size))
    y_lin = np.linspace(0, 100, int(100/cell_size))
    x_grid, y_grid = np.meshgrid(x_lin, y_lin)
    # Umweltministerium BW: für t > 10.000 Tage = 27.4 Jahre kann ein Steady state angenommen werden und das Ergebnis stimmt mit einer stationären Lösung überein
    injection_point = (120, 50)

# helper functions
def _calc_R(ne, Cw, Cs):
    return ((1-ne)*Cs+ne*Cw)/(ne*Cw)

def _calc_perm(k_cond, eta, rho_w, g):
    return k_cond*eta/(rho_w*g)

def _calc_kcond(perm, eta, rho_w, g):
    return perm*rho_w*g/eta

def _calc_vf(k_cond, grad_p):
    return -k_cond*grad_p

def _calc_va(vf, ne):
    return vf/ne

def _calc_alpha_T(alpha_L):
    return 0.1*alpha_L

def _approx_prop_of_porous_media(prop_water : float, prop_solid : float, n_e : float) -> float:
    return prop_water * n_e + prop_solid * (1-n_e) # new compared to LAHM, rule source: diss.tex

@dataclass
class Parameters:
    n_e : float = 0.25
    C_w : float = 4.2e6 # [J/m^3K]
    C_s : float = 2.4e6
    C_m : float = _approx_prop_of_porous_media(C_w, C_s, n_e)
    R : float = _calc_R(n_e, C_w, C_s) #2.7142857142857144
    rho_w : float = 1000
    rho_s : float = 2800
    g : float = 9.81
    eta : float = 1e-3
    alpha_L : float = 1 #[1,30]
    alpha_T : float = _calc_alpha_T(alpha_L)
    m_aquifer : float = 5#[5,14]

    T_gwf : float = 10.6
    T_inj_diff : float = 5
    q_inj : float = 0.00024 #[m^3/s]
    time_sim : np.array = np.array([1, 5, 27.5]) - 72/365 # 5?[years]
    # Umweltministerium BW: für t > 10.000 Tage = 27.4 Jahre kann ein Steady state angenommen werden und das Ergebnis stimmt mit einer stationären Lösung überein
    time_sim_sec : np.array = utils._time_years_to_seconds(time_sim) # [s]

    lambda_w : float = 0.65 # [-], source: diss
    lambda_s : float = 1.0 # [-], source: diss
    lambda_m : float = _approx_prop_of_porous_media(lambda_w, lambda_s, n_e)


    def __post_init__(self):
        # check second lahm requirement: energy extraction / injection must be at most 45.000 kWh/year
        energy_extraction_boundary = 45000e3/365/24 #[W] = [J/s]
        assert self.q_inj * self.C_w * self.T_inj_diff <= energy_extraction_boundary, "energy extraction must be at most 45.000 kWh/year"

@dataclass
class Testcase:
    name : str
    grad_p : float # [-]
    k_perm : float # [m^2]
    k_cond : float = 0 # [m/s]
    v_f : float = 0 # [m/s]
    v_a : float = 0 # [m/s]
    v_a_m_per_day : float = 0 # [m/day]

    def post_init(self, params:Parameters):
        self.k_cond = _calc_kcond(self.k_perm, params.eta, params.rho_w, params.g)
        self.v_f = _calc_vf(self.k_cond, self.grad_p)
        self.v_a = np.round(_calc_va(self.v_f, params.n_e), 12)
        if self.v_a < 0:
            print("v_a must be positive, I change it to its absolute value")
            self.v_a = abs(self.v_a)

        self.v_a_m_per_day = np.round(self.v_a*24*60*60, 12)

        # first lahm requirement:
        # assert self.v_a_m_per_day >= 1, "v_a must be at least 1 m per day to get a valid result"

def reverse_norm(data, stats):
    out_max = 1
    out_min = 0
    norm = stats["norm"]
    if norm == "Rescale":
        delta = stats["max"] - stats["min"]
        data = (data - out_min) / (out_max - out_min) * delta + stats["min"]
    elif norm == "Standardize":
        data = data * stats["std"] + stats["mean"]
    elif norm is None:
        pass
    else:
        raise ValueError(f"Normalization type '{stats['Norm']}' not recognized")
    
    return data


if __name__ == "__main__":
    # prior: prepare dataset with "InputParams"
    path_dataset = "/home/pelzerja/Development/datasets_prepared/1HP_NN"
    name_dataset = "benchmark_dataset_2d_100datapoints_lahm"
    path = path_dataset + "/" + name_dataset
    pathlib.Path(f"runs/{name_dataset}").mkdir(parents=True, exist_ok=True)
    info = yaml.load(open(path+"/info.yaml", "r"), Loader=yaml.SafeLoader)
    
    params = Parameters
    domain = Domain
    cell_numbers = info["CellsNumber"]
    domain.cell_size = info["CellsSize"][0]
    domain.x_lin = np.linspace(0, cell_numbers[0]*int(domain.cell_size), cell_numbers[0])
    domain.y_lin = np.linspace(0, cell_numbers[1]*int(domain.cell_size), cell_numbers[1])
    domain.x_grid, domain.y_grid = np.meshgrid(domain.x_lin, domain.y_lin)

    # read datapoint:
    for name_dp in ["RUN_0", "RUN_1", "RUN_2", "RUN_3", "RUN_4", "RUN_5", "RUN_6", "RUN_7", "RUN_8", "RUN_9"]:
    # for name_dp in ["RUN_6"]:
        inputs = np.array(torch.load(f"{path}/InputParams/{name_dp}.pt"))
        label = np.array(torch.load(f"{path}/Labels/{name_dp}.pt"))[0].T
        label = reverse_norm(label, info["Labels"]["Temperature [C]"])
        inj_point = (np.where(label == label.max()) * np.array(domain.cell_size)).squeeze()
        # if 2 inj_points are max, take the one with the smaller x-value
        if inj_point.ndim > 1:
            inj_point = inj_point[:,0]
        domain.injection_point = inj_point[::-1] + domain.cell_size/2
        testcase = Testcase(name_dp, grad_p = inputs[3], k_perm = inputs[2])
        testcase.post_init(params)

        results = {}
        time = params.time_sim_sec[2]
        delta_T_grid = params.T_gwf + lahm.delta_T(domain.x_grid-domain.injection_point[0], domain.y_grid-domain.injection_point[1], time, params, testcase)
        results["Calculated"] = delta_T_grid
        results["Label"] = label
        results["Error"] = abs(delta_T_grid - label)
        print(testcase.name, testcase.v_a_m_per_day, testcase.v_a)

        utils.plot_lahm_from_InputParams(results, filename=f"runs/{name_dataset}/{testcase.name}_lahm_isolines")