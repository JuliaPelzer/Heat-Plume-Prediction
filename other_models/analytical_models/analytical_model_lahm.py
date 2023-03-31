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

    radial_distance = _radial_distance(x, y, alpha_L, alpha_T)
    term_numerator = q_inj * T_inj_diff
    term_denominator = 4 * n_e * M * v_a * np.sqrt(np.pi * alpha_T)
    term_exponential = np.exp((x - radial_distance) / (2 * alpha_L))
    term_sqrt = 1 / np.sqrt(radial_distance)
    term_erfc = special.erfc((radial_distance - v_a * time / R) / (2 * np.sqrt(v_a * alpha_L * time / R)))
    return term_numerator / term_denominator * term_exponential * term_sqrt * term_erfc

###### helper functions
def _radial_distance(x, y, alpha_L, alpha_T):
    return np.sqrt(x**2 + y**2*alpha_L/alpha_T)