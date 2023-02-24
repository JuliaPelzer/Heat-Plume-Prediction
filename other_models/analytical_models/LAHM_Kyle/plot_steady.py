# -*- coding: utf-8 -*-
"""
Analytical formula for groundwater thermal plumes
- reading data from GIS/csv & plotting - steady state

@author: Smajil Halilovic
"""

import matplotlib.pyplot as plt
import numpy as np
import csv
from scipy import special

# reference point (center) for the coordinate system
ref_pt_x = 679700.0
ref_pt_y = 5335550.0
# direction of groundwater flow - angle
direction = 90-31.886907632299099  # deg
direction = direction*np.pi/180 # rad

def rotation(x, y, theta):
    # rotate the coordinate system
    x_r = x*np.cos(theta) + y*np.sin(theta)
    y_r = -x*np.sin(theta) + y*np.cos(theta)
    return x_r, y_r

regions_q = {}
with open('new_data/maximum_pump_per_plot_new.csv', 'r') as file:
    reader = csv.reader(file)
    row_counter = 1
    for row in reader:
        if row_counter>1:
            id = row[0]
            q_demand = row[5]
            regions_q[id] = float(q_demand)* 10 ** (-3) # [m3/s]
        else:
            # skip first row
            row_counter=2

well_ext_ids = {}
well_inj_ids = {}
with open('new_data/well_positions10.csv', 'r') as file:
    reader = csv.reader(file)
    row_counter = 1
    x_inj = []
    y_inj = []
    x_ext = []
    y_ext = []
    q_rates = []
    for row in reader:
        if row_counter>1:
            id = row[0]
            x_coord = float(row[2])-ref_pt_x
            y_coord = float(row[3])-ref_pt_y
            # rotate the coordinate system
            x_coord, y_coord = rotation(x_coord, y_coord, direction)
            if row[1]=="extraction":
                # lists of coordinates
                x_ext.append(x_coord)
                y_ext.append(y_coord)
                # dictionary that relates regions and wells
                if id in well_ext_ids:
                    well_ext_ids[id].append(x_ext.index(x_coord))
                else:
                    well_ext_ids[id] = []
                    well_ext_ids[id].append(x_ext.index(x_coord))
            else:
                # injection wells
                # lists of coordinates
                x_inj.append(x_coord)
                y_inj.append(y_coord)
                # list of pumping rates
                q_rates.append(regions_q[id])
                # dictionary that relates regions and wells
                if id in well_inj_ids:
                    well_inj_ids[id].append(x_inj.index(x_coord))
                else:
                    well_inj_ids[id] = []
                    well_inj_ids[id].append(x_inj.index(x_coord))
        else:
            # skip first row
            row_counter=2

# Parameters needed in LAHM model:
n = 0.25  # [-]
b = 1 #8.510619493436851 # [m]
v_a = 0.5 # [m/d]
v_a = v_a / (24 * 60 * 60)
#v_darcy = 2.9 * 10 ** (-7) # [m/s]
#v_a = v_darcy
alpha_L = 1.8 # [m]
alpha_T = 0.1*alpha_L # 0.18 #0.18  # [m]
Cw = 4.185 * 10 ** 6  # [J/Km**3]
Cm = 2.888 * 10 ** 6  # [J/Km**3]
R = Cm / (n * Cw)  # [-]
DT_inj = 5  # [K]

def delta_temp(x, y, q_inj, t):
    # Analytical formula for temperature change
    r = np.sqrt(x ** 2 + (y ** 2) * (alpha_L / alpha_T))
    ampl_term = 1 / (4 * n * b * v_a * np.sqrt(np.pi * alpha_T))
    exp_term = np.exp((x - r) / (2 * alpha_L))
    erfc_term = special.erfc((r - v_a * t / R) / (2 * np.sqrt(v_a * alpha_L * t / R)))
    delta_T = q_inj * DT_inj * ampl_term * exp_term * (1 / np.sqrt(r)) * erfc_term

    return delta_T


# Current time [s] - should be long enough for the steady state
t_days = 365 # days
t_current = t_days*24*60*60 # convert to seconds

## Plotting

# # plotting location of all possible wells
# fig3 = plt.figure(1)
# for i in range(len(x_inj)):
#     plt.scatter((x_inj[i]), (y_inj[i]), marker="x", color="black", s=15)
# # plt.xlim([-5, 100])
# # plt.ylim([0, 50])
# for i in range(len(x_ext)):
#     plt.scatter((x_ext[i]), (y_ext[i]), marker="o", color="black", s=15)
# plt.xlabel('$x$ [m]'); plt.ylabel('$y$ [m]')
# plt.show()

# plotting temperature field for all HPs
x_lim = np.arange(406, 534, 0.5)
y_lim = np.arange(-361, -233, 0.5)
x_grid, y_grid = np.meshgrid(x_lim, y_lim)
Delta_T = 0*x_grid
for i in range(1):#range(len(x_inj)):
    q_inj = 0.01 * 0.001
    Delta_T += delta_temp(x_grid-x_inj[i], y_grid-y_inj[i], q_inj, t_current) #q_rates[i]


fig = plt.figure(3)
ax = fig.add_subplot(111)
T_0 = 10

print("delta T at injection point", T_0+Delta_T[int((x_inj[0]-x_lim[0])/0.5), int((y_inj[0]-y_lim[0])/0.5)])


levels = np.arange(10, 15.0, 0.5)
cax = ax.contourf(x_grid, y_grid, T_0+Delta_T,
                  levels=levels, extend='both',
                  cmap='RdBu_r')
cbar = fig.colorbar(cax)
ax.set_xlabel('$x$ [m]'); ax.set_ylabel('$y$ [m]')
plt.show()
