import analytical_model_lahm as lahm
import benchmark_analytical as bm_lahm
import matplotlib.pyplot as plt
import numpy as np
from scipy import special

# # Implementation of LAHM model and application to my dataset

# only applicable to:
# - velocities over 1m/day; good for v >= 10m/day
# - energy extraction under 45.000 kWh/year

# steady state model (Willibald, 1980):
# - calculates temperature field
# - with continous point source
# - convective and dispersive heat transport
# - Annahmen:
#     - konstante mit Grundwasser erfüllte Mächtigkeit
#     - einheitliches Grundwassergefälle
#     - konstante Permeabilität
#     - konstanter transportwirksamer Hohlraumanteil
#     - unbeeinflusste, konstante Hintergrundtemperatur
# - -> parallele Grundsträmung mit konstanter Filtergeschwindigkeit
# - Vernachlässigung:
#     - diseser Strömung durch Injection/extraction well (bzgl. Höhe Grundwasserstand, Änderung Strömungsrichtung, -geschwindigkeit)
#     - Austausch mit Atmosphäre
#     - konduktive Prozesse


# - equations:
# $$ v_f = k_f * I$$
# $$ v_a = v_f / n $$
# $$ x_0 = \frac{1}{4 \cdot \pi \cdot \alpha_T}(\frac{Q \cdot \Delta T_E}{m \cdot v_f \cdot \Delta T})^2$$
# $$ y = +- \sqrt{4 \cdot \alpha_T \cdot x \cdot ln(\frac{Q \cdot \Delta T_E}{m \cdot v_f \cdot \Delta T \sqrt{4 \cdot \pi \cdot \alpha_T \cdot x}})}$$

# - $v_f$ : Filtergeschwindigkeit [m/s]
# - $k_f$ : Permeabilität des Grundwasserleiters [m/s] = k_perm
# - $I$ : Gradient der Piezometerhöhe, Grundwassergefälle[-] = grad_p?
# - $v_a$ : mittlere Abstandsgeschwindigkeit [m/s] (matches with LAHM)
# - $n$ : durchflusswirksamer Hohlraumanteil [–] (=effektive Porosität) (matches with LAHM-$n_e$- effektive Porosität)
# - $x_0$ : Abstand der gesuchten Isotherme vom Infiltrationsbrunnen auf der Stromlinie im Abstrom des Rückgabebrunnens [m] (nicht zu verwechseln mit dem Staupunkt oder der unterstromigen Reichweite der Anströmung eines Brunnens, die oft auch mit $x_0$ bezeichnet werden)
# - $x$ : Abstand zum Rückgabebrunnen auf der Stromlinie im Abstrom des Rückgabebrunnens [m]
# - $y$ : seitliche Ausdehnung der Isotherme, berechnet als seitlicher Abstand zur Stromlinie für Punkte $x < x_0$; bei $x_0$ ist $y = 0$; [m]
# - $\alpha_T$ : Querdispersivität [m] (matches with LAHM)
# - $Q$ : Infiltrationsrate [m³ s-1] (matches with LAHM)
# - $m$ : Grundwassererfüllte Mächtigkeit [m] (matches with LAHM-$M$)
# - $\Delta T_E$ : Unterschied zwischen Einleittemperatur und unbeeinflusster Grundwassertemperatur [K] (matches with LAHM-$T_{inj}$)
# - $\Delta T$ : Gesuchte Isotherme als Differenz zur Grundwassertemperatur [K] (matches with LAHM) (vorgebbar über welche Isolinien interessieren mich)

def position_of_isoline(x, y, q_inj, T_isoline, parameters):
    """
    Calculates the position of the isoline for a given temperature at steady state.
    """
    # constants
    PI = np.pi
    kf = parameters["k_perm"]
    grad_p = parameters["grad_p"] # former: "I"
    ne = parameters["n_e"]
    alpha_T = parameters["alpha_T"]
    m = parameters["M"]
    delta_T = parameters["T_inj_diff"]

    # calc v_f
    vf = kf * grad_p # weird..?
    # calc v_a
    va = vf / ne
    print(x.shape, y.shape)
    # calc prefactor
    term_alpha_factor = 4 * PI * alpha_T
    term_qT_factor = q_inj * delta_T / (m * vf * T_isoline)
    x_0 = 1 / term_alpha_factor * term_qT_factor**2
    y = np.sqrt(term_alpha_factor * x * np.log(term_qT_factor * 1/np.sqrt(term_alpha_factor*x)))
    y_minus = - y
    print(x_0, y.shape, y_minus.shape)
    print("BROKEN - x vs. y? position - how large range?...")
    return x_0, y, y_minus

def plot_isoline(x, y, y_minus):
    """
    Plots the isoline for a given temperature at steady state.
    """
    plt.plot(x, y, color="blue")
    plt.plot(x, y_minus, color="blue")