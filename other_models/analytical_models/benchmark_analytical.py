import numpy as np
from dataclasses import dataclass
import analytical_model_lahm as lahm
import analytical_model_pahm as pahm
import analytical_steady_state_model_willibald as willibald
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
    k_cond : float # [m/s]
    k_perm : float = 0 # [m^2]
    v_f : float = 0 # [m/s]
    v_a : float = 0 # [m/s]
    v_a_m_per_day : float = 0 # [m/day]

    def post_init(self, params:Parameters):
        self.k_perm = _calc_perm(self.k_cond, params.eta, params.rho_w, params.g)
        self.v_f = _calc_vf(self.k_cond, self.grad_p)
        self.v_a = _calc_va(self.v_f, params.n_e)
        self.v_a_m_per_day = self.v_a*24*60*60

        # first lahm requirement: 
        # assert self.v_a_m_per_day <= -1, "v_a must be at least 1 m per day to get a valid result"
    
def calc_and_plot(domain:Domain, params:Parameters, testcase:Testcase, approach:str = "lahm"):
    results = {}
    for t_idx, time in enumerate(params.time_sim_sec):
        if approach == "lahm":
            delta_T_grid = lahm.delta_T(domain.x_grid-domain.injection_point[0], domain.y_grid-domain.injection_point[1], time, params, testcase)
            results[f"{np.round(params.time_sim[t_idx],2)} years"] = delta_T_grid+params.T_gwf

        elif approach == "pahm":
            delta_T_grid = pahm.delta_T(domain.x_grid, domain.y_grid-domain.injection_point[1], time, params, testcase)
            flow = np.full_like(delta_T_grid, params.T_gwf)
            flow[:,domain.injection_point[0]:] += delta_T_grid[:,0:-domain.injection_point[0]]
            results[f"{np.round(params.time_sim[t_idx],2)} years"] = flow
        # results[f"{np.round(params.time_sim[t_idx],2)} years"] = delta_T_grid+params.T_gwf
        print("T diff", delta_T_grid[int(domain.injection_point[1]/domain.cell_size), int(domain.injection_point[0]/domain.cell_size)])

    utils.plot_temperature_field(results, domain.x_grid, domain.y_grid, filename=f"{approach}_{testcase.name}_combined", params=params)
    utils.plot_temperature_field(results, domain.x_grid, domain.y_grid, filename=f"{approach}_{testcase.name}_isolines")

def run_willibald(domain, parameters:dict):
    T_isoline = 12
    x_isoline, y_isoline, y_minus_isoline = willibald.position_of_isoline(domain.x_lin+1, domain.y_lin+1, parameters["q_inj"], T_isoline, parameters)  
    # x, y, y_minus = willibald.position_of_isoline(domain.x_grid-domain.injection_point[1], domain.y_grid-domain.injection_point[0], parameters["q_inj"], T_isoline, parameters)  
    willibald.plot_isoline(domain.x_lin+x_isoline, y_isoline, y_minus_isoline)

if __name__ == "__main__":

    params = Parameters
    domain = Domain

    testcases = [
        Testcase(f'benchmark_testcase_0', grad_p = 0.0015, k_cond = 0.0001),
        Testcase(f'benchmark_testcase_1', grad_p = 0.0015, k_cond = 0.002),
        Testcase(f'benchmark_testcase_2', grad_p = 0.003, k_cond = 0.01),
        Testcase(f'benchmark_testcase_3', grad_p = 0.0035, k_cond = 0.05)
    ]
    for testcase in testcases:
        testcase.post_init(params)
        # print(testcase)

    approach = "lahm"

    for testcase in testcases:
        calc_and_plot(domain, params, testcase, approach)
        # x, y, y_minus = run_willibald(domain, parameters)