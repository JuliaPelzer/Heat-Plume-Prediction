# Implementation of LAHM model and application to my dataset
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from copy import copy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Dict

# # PAHM model 
# source: Analytical solutions for predicting thermal plumes of groundwater heat pump systems, Pophillat et al (2020)

# PAHM = 2D solute transport analytical solution for transient conditions and homogeneous parallel groundwater flow, assuming a continuous and finite planar source
# - transformed equation for heat transport is described in Hähnlein et al.
# - for a semi-infinite domain -> ignores zone up-gradient from the source location

# exists for 3D too - check Hähnlein et al. for details

# applicable to:
# - intermediate groundwater flow velocities: 1 m/d < v < 10 m/d
# - transient conditions and homogeneous parallel groundwater flow, assuming a continuous and finite planar source

# f0 : energy injection per length of the borehole [W/m]
# Y : dimension of source in y direction
# qh : injected heat power
# d_x, d_y : longitudinal and transverse hydrodynamic dispersion coefficients (L/T: with respect to the groundwater flow direction)
# ymax : maximum (downgradient) width of steady-state plume
# lamda_m : thermal conductivity of porous media [W/(mK)] = [J/(msK)]

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
    Cw = parameters.C_w
    lambda_m = parameters.lambda_m
    v_a = testcase.v_a

    if v_a < 0:
        print("v_a must be positive, I change it to its absolute value")
        v_a = abs(v_a)

    ymax = q_inj / (M * v_a * n_e)
    # y0 = q_inj / (2 * M * v_a * n_e)
    d_x = lambda_m / (n_e * Cw) + alpha_L * v_a
    d_y = lambda_m / (n_e * Cw) + alpha_T * v_a
    qh = T_inj_diff * Cw * q_inj
    f0 = qh / M
    delta_T0 = f0 / (v_a * n_e * Cw * ymax)

    prefactor = delta_T0 / 4.0
    inner_erfc = (R * x - v_a * time)/(2 * np.sqrt(d_x * R * time))
    def inner_erf(ymax):
        return (y + ymax/2)/(2 * np.sqrt(d_y * x / v_a))
    inner_erf_plus = inner_erf(ymax)
    inner_erf_minus = inner_erf(-ymax)
    return prefactor * special.erfc(inner_erfc) * (special.erf(inner_erf_plus) - special.erf(inner_erf_minus))