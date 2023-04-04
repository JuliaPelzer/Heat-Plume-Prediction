import numpy as np
from scipy import special
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from copy import copy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Dict

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
            levels = [params.T_gwf, params.T_gwf + 1, params.T_gwf + params.T_inj_diff]
            CS = plt.contour(x_grid, y_grid, value, levels=levels, cmap='Pastel1', extent=(0,1280,100,0))
            plt.clabel(CS, inline=1, fontsize=10)
            plt.imshow(value, cmap="RdBu_r", extent=(0,1280,100,0))
            plt.gca().invert_yaxis()
        else:
            levels = np.arange(10.6, 15.6, 0.25)
            plt.contourf(x_grid, y_grid, value, levels=levels, cmap='RdBu_r', extent=(0,1280,100,0))
        plt.ylabel("x [m]")
        _aligned_colorbar(label="Temperature [°C]")

    plt.xlabel("y [m]")
    # plt.show()
    plt.savefig(f"{filename}.png")
    plt.savefig(f"{filename}.svg")

def plot_different_versions_of_temperature(data, x_grid, y_grid, title="", ellipses=None):
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

def _aligned_colorbar(*args,**kwargs):
    cax = make_axes_locatable(plt.gca()).append_axes("right",size= 0.3,pad= 0.05)
    plt.colorbar(*args,cax=cax,**kwargs)