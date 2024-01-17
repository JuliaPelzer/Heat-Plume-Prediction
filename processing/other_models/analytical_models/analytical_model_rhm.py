# Implementation of LAHM model and application to my dataset
from copy import copy
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import special

# # RHM model

# - source: analytical solutions for predicting thermal plumes of groundwater heat pump systems, Pophillat et al (2020)
# - Guimerà et al. modified the 2D solute transport analytical solution for heat transport simulations and transient conditions, original given by Gelhar and Collins for contamination
# - which estimates the contaminant distribution in a 

# - homogeneous confined aquifer
# - fully penetrating injection well 
# - assuming a continuous line-source  
# - no background groundwater flow 

def delta_T(x, y, t, params):
    R = params.R
    q_inj = params.q_inj
    M = params.m_aquifer
    n_e = params.n_e
    alpha_L = params.alpha_L
    lambda_m = params.lambda_m
    C_m = params.C_m

    r = np.sqrt(x**2 + y**2)
    A_T = 1 / R * q_inj / (2 * np.pi * n_e * M)
    r_star = np.sqrt(2*A_T * t)
    upper_term = r**2 - r_star**2

    lower_term_sqrt = 4/3 * alpha_L * r_star**3 + lambda_m / (A_T * C_m) * r_star**4
    return 0.5 * special.erfc(upper_term / (2*np.sqrt(lower_term_sqrt)))