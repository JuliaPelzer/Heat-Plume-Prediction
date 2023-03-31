# Implementation of LAHM model and application to my dataset
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from copy import copy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Dict

###### Requirements etc
# only applicable to:
# - velocities over 1m/day
# - energy extraction under 45.000 kWh/year

# LAHM model (Kinzelbach, 1987):
# - calculates temperature field
# - with continous point source
# - convective and dispersive heat transport
# - in a homogeneous confined aquifer ("homogener gespannter Porengrundwasserleiter")
# - under instationary conditions

# - equations:
# $$ \Delta T (x,y,t) = \frac{Q \cdot \Delta T_{inj}}{4 \cdot n_e \cdot M \cdot v_a \cdot \sqrt{\pi \cdot \alpha_T}}
# \cdot \exp{(\frac{x - r}{2 \cdot \alpha_L})} \cdot \frac{1}{\sqrt{r}}
# \cdot erfc(\frac{r - v_a \cdot t/R}{2 \cdot \sqrt{v_a \cdot \alpha_L \cdot t/R}}) 
# $$
# with:
# $$ r = \sqrt{x^2 + y^2 \cdot \frac{\alpha_L}{\alpha_T}} $$

# - $\Delta T$ : Gesuchte Isotherme als Differenz zur unbeeinflussten Grundwassertemperatur [K]
# - $Q$ : Injektionsrate [m 3 /s]
# - $T_{inj}$ : Differenz zwischen der Injektionstemperatur und der unbeeinflussten Grundwassertemperatur [K]
# - $n_e$ : effektive Porosität [–]
# - $M$ : genutzte, grundwassererfüllte Mächtigkeit [m]
# - $v_a$ : Abstandsgeschwindigkeit [m/s]
# - $α_{L,T}$ : Längs- und Querdispersivität [m]
# - $x$, $y$: Längs- und Querkoordinaten [m]
# - $t$: Zeit [s]
# - $R$: Retardation [–]
# - $r$: radialer Abstand vom Injektionsbrunnen [m]

######## Next steps
# TODO temperature added too high

# read input from file
# test on dataset
# adaptation to 3D
# read streamlines
# coordination transformation

def delta_T(x, y, time, parameters, testcase):
    """
    Calculate the temperature difference between the injection well and the point (x, y) at time t.
    """
    
    n_e = parameters.n_e
    M = parameters.m_aquifer
    alpha_L = parameters.alpha_L
    alpha_T = parameters.alpha_T
    R = parameters.R
    T_inj_diff = parameters.T_inj_diff
    q_inj = parameters.q_inj
    v_a = testcase.v_a

    if v_a < 0:
        print("v_a must be positive, I change it to its absolute value")
        v_a = abs(v_a)

    radial_distance = _radial_distance(x, y, alpha_L, alpha_T)
    term_numerator = q_inj * T_inj_diff
    term_denominator = 4 * n_e * M * v_a * np.sqrt(np.pi * alpha_T)
    term_exponential = np.exp((x - radial_distance) / (2 * alpha_L))
    term_sqrt = 1 / np.sqrt(radial_distance)
    term_erfc = special.erfc((radial_distance - v_a * time / R) / (2 * np.sqrt(v_a * alpha_L * time / R)))
    return term_numerator / term_denominator * term_exponential * term_sqrt * term_erfc

def ellipse_10_percent(inj_point, alpha_L, alpha_T):
    height = 4 * np.sqrt(alpha_L*alpha_T)
    width = 4 * alpha_L
    return Ellipse(inj_point, width, height, fill=False, color="red")

def ellipse_1_percent(inj_point, alpha_L, alpha_T):
    height = 40 * np.sqrt(alpha_L*alpha_T)
    width = 40 * alpha_L
    return Ellipse(inj_point, width, height, fill=False, color="red")

def plot_temperature_field(data:Dict, x_grid, y_grid, filename="", params=None):
    """
    Plot the temperature field.
    """
    n_subplots = len(data.keys())
    _, axes = plt.subplots(n_subplots,1,sharex=True,figsize=(38.4,3*(n_subplots)))
    
    for index, (key, value) in enumerate(data.items()):
        plt.sca(axes[index])
        plt.title(f"{key}")
        if params:
            plt.gca().invert_yaxis()
            levels = [params.T_gwf, params.T_gwf + 1, params.T_gwf + params.T_inj_diff]
            CS = plt.contour(x_grid, y_grid, value, levels=levels, cmap='Pastel1', extent=(0,1280,100,0))
            plt.clabel(CS, inline=1, fontsize=10)
            plt.imshow(value, cmap="RdBu_r")
        else:
            levels = np.arange(10.6, 15.6, 0.25)
            plt.contourf(x_grid, y_grid, value, levels=levels, cmap='RdBu_r', extent=(0,1280,100,0))
        plt.ylabel("x [m]")
        _aligned_colorbar(label="Temperature [°C]")

    plt.xlabel("y [m]")
    # plt.show()
    plt.savefig(f"{filename}.png")

def plot_different_versions_of_temperature_lahm(data, x_grid, y_grid, title="", ellipses=None):
    """
    Plot the temperature field.
    """
    n_subplots = 3
    _, axes = plt.subplots(n_subplots,1,sharex=True,figsize=(20,3*(n_subplots)))
    
    for index in range(n_subplots):
        plt.sca(axes[index])
        if index == 0:
            plt.title("Temperature field")
            plt.imshow(data,extent=(0,1280,100,0))
            plt.gca().invert_yaxis()
        elif index == 1:
            plt.title("Temperature field with contour lines")
            plt.contourf(x_grid, y_grid, data, extent=(0,1280,100,0))
        elif index == 2:
            plt.title("Temperature field with focused contour lines [10 °C, 15 °C]")
            levels = np.arange(10, 15.0, 0.25)
            plt.contourf(x_grid, y_grid, data, levels=levels, cmap='RdBu_r', extent=(0,1280,100,0))

        if ellipses:
            for ellipse in ellipses:
                plt.gca().add_patch(copy(ellipse))
                plt.plot(ellipse.center[0], ellipse.center[1], "ro")
                plt.plot(120, 50, "g+")

        plt.ylabel("x [m]")
        _aligned_colorbar(label=title)
    plt.xlabel("y [m]")
    # plt.show()
    plt.savefig(f"{title}.png")

###### helper functions
def _time_years_to_seconds(time_years):
    factor = 365 * 24 * 60 * 60
    return time_years * factor

def _velocity_m_day_to_m_s(velocity_m_day):
    return velocity_m_day / (24 * 60 * 60)

def _radial_distance(x, y, alpha_L, alpha_T):
    return np.sqrt(x**2 + y**2*alpha_L/alpha_T)

def _aligned_colorbar(*args,**kwargs):
    cax = make_axes_locatable(plt.gca()).append_axes("right",size= 0.3,pad= 0.05)
    plt.colorbar(*args,cax=cax,**kwargs)
    
if __name__ == "__main__":

    ###### parameters
    n_e = 0.25 # effective porosity [-], same value as in pflotran
    M = 5.0 # aquifer thickness / Grundwassermächtigkeit [m], source: Smajil
    v_a_m_per_day = 1 #requirement min 1-10 [m/d], Smajil 0.5
    alpha = 1.8 # [m], source: Kyle TODO Vary 
    Cw = 4.185 * 10 ** 6  # [J/Km**3], source: Smajil
    Cm = 2.888 * 10 ** 6  # [J/Km**3], source: Smajil
    R = Cm / (n_e * Cw)  # [-], source: Smajil
    T_inj_diff = 5  # [K], same value as in pflotran
    
    assert v_a_m_per_day >= 1, "v_a_m_per_day must be greater than 1 m/d to get plausible results"
    if v_a_m_per_day <=10:
        method = "PAHM"
        print("Method should be PAHM but not implemented") #TODO
        # but method is developed for semi-infinite domain and ignores region up-gradient of injection point; TODO: extend or make LAHM for this region and get non-continuous result 
    else:
        method = "LAHM"

    parameters = {
        "n_e": n_e, 
        "M": M, 
        "v_a": _velocity_m_day_to_m_s(v_a_m_per_day), # [m/s]
        "alpha_L": alpha, # [m], source: Smajil
        "alpha_T": 0.1*alpha, # [m], source: Smajil
        "R": R, 
        "T_inj_diff": T_inj_diff,
        "T_gwf": 10.6, # [K], 
    }

    cell_size = 1
    x_lin = np.linspace(0, 1280, int(1280/cell_size))
    y_lin = np.linspace(0, 100, int(100/cell_size))
    x_grid, y_grid = np.meshgrid(x_lin, y_lin)
    time_steady_state = _time_years_to_seconds(10) # [s] TODO research LAHM + time
    # Umweltministerium BW: für t > 10.000 Tage = 27.4 Jahre kann ein Steady state angenommen werden und das Ergebnis stimmt mit einer stationären Lösung überein
    injection_point = (120, 50)
    q_inj = 1e-5#*(cell_size**2) #4.86*10**-5 # [m**3/s], same value as in pflotran, Kyle: 1*10^-5

    delta_T_grid = delta_T(x_grid-injection_point[0], y_grid-injection_point[1], time_steady_state, q_inj, parameters)
    print("T diff", delta_T_grid[int(injection_point[1]/cell_size), int(injection_point[0]/cell_size)])

    ellipse_10 = ellipse_10_percent(injection_point, parameters["alpha_L"], parameters["alpha_T"])
    # ellipse_1 = ellipse_1_percent(injection_point, parameters["alpha_L"], parameters["alpha_T"])
    plot_temperature_field({"10 years": delta_T_grid+parameters["T_gwf"]}, x_grid, y_grid, filename="test_temperature_plot") #, ellipses=[ellipse_10])