import numpy as np
from dataclasses import dataclass
import analytical_temperature_prediction_lahm as lahm
import analytical_steady_state_model_willibald as willibald

def check_lahm_requirements(parameters):
    # first lahm requirement: 
    print(parameters["v_a"]*60*60*24)
    assert parameters["v_a"]*60*60*24 <= -1, "v_a must be at least 1 m per day to get a valid result"
    # second lahm requirement: energy extraction / injection must be at most 45.000 kWh/year
    energy_extraction_boundary = 45000e3/365/24 #[W] = [J/s]
    print(parameters["q_inj"]*1000)
    print(parameters["q_inj"] * parameters["C_w"] * parameters["T_inj_diff"], energy_extraction_boundary)
    assert parameters["q_inj"] * parameters["C_w"] * parameters["T_inj_diff"] <= energy_extraction_boundary, "energy extraction must be at most 45.000 kWh/year"

def calc_and_plot(domain, parameters):
    delta_T_grid = lahm.delta_T(domain.x_grid-domain.injection_point[0], domain.y_grid-domain.injection_point[1], parameters["time_sim"], parameters["q_inj"], parameters)
    print("T diff", delta_T_grid[int(domain.injection_point[1]/domain.cell_size), int(domain.injection_point[0]/domain.cell_size)])

    ellipse_10 = lahm.ellipse_10_percent(domain.injection_point, parameters["alpha_L"], parameters["alpha_T"])
    lahm.plot_temperature_lahm(delta_T_grid+parameters["T_gwf"], domain.x_grid, domain.y_grid, title=parameters["name"], ellipses=[ellipse_10])

def run_willibald(domain, parameters:dict):
    T_isoline = 12
    x_isoline, y_isoline, y_minus_isoline = willibald.position_of_isoline(domain.x_lin+1, domain.y_lin+1, parameters["q_inj"], T_isoline, parameters)  
    # x, y, y_minus = willibald.position_of_isoline(domain.x_grid-domain.injection_point[1], domain.y_grid-domain.injection_point[0], parameters["q_inj"], T_isoline, parameters)  
    willibald.plot_isoline(domain.x_lin+x_isoline, y_isoline, y_minus_isoline)

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

if __name__ == "__main__":
    # default values (diss.tex)
    Cw = 4.2e6 # [J/m^3K]
    Cs = 2.4e6
    ne = 0.25
    R = _calc_R(ne, Cw, Cs) #2.7142857142857144
    rho_w = 1000
    rho_s = 2800
    g = 9.81
    eta = 1e-3
    alpha_L = 1 #[1,30]
    alpha_T = _calc_alpha_T(alpha_L)
    m_aquifer = 5#[5,14]

    T_gwf = 10.6
    T_inj_diff = 5
    q_inj = 0.00024 #[m^3/s]
    t_sim = 5 # [years]

    testcases = [
        {"case": "1", "grad_p": 0.0015, "k_cond": 0.002, "k_perm": 0, "vf": 0, "va": 0, "va_m_per_day": 0},
        {"case": "2", "grad_p": 0.003, "k_cond": 0.01, "k_perm": 0, "vf": 0, "va": 0, "va_m_per_day": 0},
        {"case": "3", "grad_p": 0.0035, "k_cond": 0.05, "k_perm": 0, "vf": 0, "va": 0, "va_m_per_day": 0}
    ]

    for testcase in testcases:
        k_perm = _calc_perm(testcase["k_cond"], eta, rho_w, g)
        vf = _calc_vf(testcase["k_cond"], testcase["grad_p"])
        va = _calc_va(vf, ne)
        testcase["k_perm"] = k_perm
        testcase["vf"] = vf
        testcase["va"] = va
        testcase["va_m_per_day"] = va*60*60*24

    domain = Domain

    for testcase in testcases:

        parameters = {
            "name": f'benchmark_testcase_{testcase["case"]}_after_{t_sim}_years',
            "C_w": Cw,
            "n_e": ne, 
            "M": m_aquifer, 
            "v_a": testcase["va"], # [m/s]
            "alpha_L": alpha_L,
            "alpha_T": alpha_T,
            "R": R, 
            "T_inj_diff": T_inj_diff,
            "T_gwf": 10.6, # [K], 
            "q_inj": q_inj, # [m^3/s]
            "time_sim": lahm._time_years_to_seconds(t_sim), # [s]
            "k_perm": testcase["k_perm"], # [m^2] # for Willibald
            "grad_p": testcase["grad_p"], # [m/s] # for Willibald
        }
        check_lahm_requirements(parameters)
        calc_and_plot(domain, parameters)
        # x, y, y_minus = run_willibald(domain, parameters)