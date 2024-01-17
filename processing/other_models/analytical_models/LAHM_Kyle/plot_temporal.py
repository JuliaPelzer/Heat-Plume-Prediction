# -*- coding: utf-8 -*-
"""
Analytical formula for groundwater thermal plumes
- reading data from GIS/csv & plotting - dynamic case

@author: Smajil Halilovic
"""

import csv

import matplotlib.pyplot as plt
import numpy as np
from scipy import special

# reference point (center) for the coordinate system
ref_pt_x = 679700.0
ref_pt_y = 5335550.0
# direction of groundwater flow - angle
direction = 90-31.886907632299099  # deg
direction = direction*np.pi/180 # rad
# conversion factor days to seconds:
day_to_sec = 24*60*60

def rotation(x, y, theta):
    # rotate the coordinate system
    x_r = x*np.cos(theta) + y*np.sin(theta)
    y_r = -x*np.sin(theta) + y*np.cos(theta)
    return x_r, y_r

with open('lahm_real_domain/maximum_pump_per_plot.csv', 'r') as file:
    reader = csv.reader(file)
    bgs_ids = []
    inj_well_ids = []
    inj_x_coords = []
    inj_y_coords = []
    ext_well_ids = []
    ext_x_coords = []
    ext_y_coords = []
    #bg_q_rates = []
    row_counter = 1
    for row in reader:
        if row_counter>1:
            id = row[0]
            inj_id = row[1]
            inj_x = row[2]
            inj_y = row[3]
            ext_id = row[4]
            ext_x = row[5]
            ext_y = row[6]
            #q_demand = row[13]
            bgs_ids.append(id)
            inj_well_ids.append(int(inj_id))
            inj_x_coords.append(float(inj_x)-ref_pt_x)
            inj_y_coords.append(float(inj_y)-ref_pt_y)
            ext_well_ids.append(int(ext_id))
            ext_x_coords.append(float(ext_x)-ref_pt_x)
            ext_y_coords.append(float(ext_y)-ref_pt_y)
            #bg_q_rates.append(float(q_demand))
        else:
            # skip first row
            row_counter=2

# read time-series demand (pumping rates) from multiple csv files
q_rates_all = [] # list of lists,
# where each element correspond to one building/region with a timeseries (list) of pumping rates
for id in bgs_ids:
    id_str = id.split("/")
    file_str = id_str[0] + "_" + id_str[1] + "_" + id_str[2]
    file_path = "lahm_real_domain/load_time_series/" + file_str + ".csv"
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        bg_q_rates = []
        row_counter = 1
        for row in reader:
            if row_counter > 1:
                q_demand = row[2]
                bg_q_rates.append(float(q_demand)* 10 ** (-3))
            else:
                # skip first row
                row_counter = 2
    q_rates_all.append(bg_q_rates)

# rotate the coordinate system:
for i in range(len(inj_x_coords)):
    inj_x_coords[i], inj_y_coords[i] = rotation(inj_x_coords[i], inj_y_coords[i], direction)
for i in range(len(ext_x_coords)):
    ext_x_coords[i], ext_y_coords[i] = rotation(ext_x_coords[i], ext_y_coords[i], direction)

# Parameters needed in LAHM model:
n = 0.3  # [-]
b = 8.510619493436851 # [m]
#v_a = 10  # [m/d]
#v_a = v_a / (24 * 60 * 60)
v_darcy = 4 * 10 ** (-5) # [m/s]
v_a = v_darcy
alpha_L = 1.8  # [m]
alpha_T = 0.18  # [m]
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

# Regions
# = {1: (1,2), 2: [3], 3: [4], 4: (5,6)}
regions_ext = {}
for i in range(len(bgs_ids)):
    regions_ext[i] = [i]
#regions_inj = {1: (1,2), 2: [3], 3: [4], 4: (5,6)}
regions_inj = {}
for i in range(len(bgs_ids)):
    regions_inj[i] = [i]

# Generating impulses for pumping rates (temporal superposition)
Qinj_wells = []
for i in range(len(q_rates_all)):
    q_aux = []
    for t in range(len(q_rates_all[i])):
        q_aux.append(q_rates_all[i][t])
    Qinj_wells.append(q_aux)
for i in range(len(q_rates_all)):
    for t in range(1,len(q_rates_all[i])):
        Qinj_wells[i][t] = q_rates_all[i][t] - q_rates_all[i][t-1]
#Qinj_region = {1: Qinj_1, 2: Qinj_2}
#Qinj_wells = {1: Qinj_1, 2: Qinj_1, 3: Qinj_2, 4: Qinj_2}

# Coordinates of wells:
xi = inj_x_coords
yi = inj_y_coords
xe = ext_x_coords
ye = ext_y_coords

Inj = range(len(xi))
Ext = range(len(xe))

# Time steps [s] - list
t_days = 365
t_end = t_days*24*60*60
t_months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31] # nr. of days in months
t_current = []
t_aux = 0
for i in range(len(t_months)):
    t_current.append(t_aux)
    t_aux += t_months[i]
#t_current = [0, 31, 31+28, 31+28+31, 31+28+31+30, 31+28+31+30+31, 31+28+31+30+31+30,
#             31+28+31+30+31+30+31, 31+28+31+30+31+30+31+31, 31+28+31+30+31+30+31+31+30,
#             31+28+31+30+31+30+31+31+30+31, 31+28+31+30+31+30+31+31+30+31+30]
t_list = [t_end-t*day_to_sec for t in t_current]

# Plotting

# plotting location of all possible wells
fig3 = plt.figure(1)
for i in range(len(xi)):
    plt.scatter((xi[i]), (yi[i]), marker="x", color="black", s=15)
for i in range(len(xe)):
    plt.scatter((xe[i]), (ye[i]), marker="o", color="black", s=15)
plt.xlabel('$x$ [m]'); plt.ylabel('$y$ [m]')
plt.show()

# plotting temperature field for given well locations
x_lim = np.arange(250, 900, 0.25)
y_lim = np.arange(-550, 200, 0.25)
x_grid, y_grid = np.meshgrid(x_lim, y_lim)
# one plot at the end of each month:
t_end_month = []
for i in range(len(t_months)):
    # generate list of timesteps at the end of each month:
    if i==0:
        t_end_month.append(t_months[i]*day_to_sec)
    else:
        t_end_month = [t_end_month[k] + t_months[i]*day_to_sec for k in range(len(t_end_month))]
        t_end_month.append(t_months[i]*day_to_sec)
    # compute Delta_T (temperature field changes) - at the end of each month:
    Delta_T = 0 * x_grid
    for t in range(len(t_end_month)):
        for j in range(10):#Inj:
            Delta_T += delta_temp(x_grid - xi[j], y_grid - yi[j], Qinj_wells[j][t], t_end_month[t])
    # generate plot - at the end of each month
    fig = plt.figure(i+3)
    ax = fig.add_subplot(111)
    T_0 = 10
    levels = np.arange(5, 10.5, 0.5)
    cax = ax.contourf(x_grid, y_grid, T_0 - Delta_T,
                      levels=levels, extend='both',
                      cmap='RdYlBu_r')
    cbar = fig.colorbar(cax)
    ax.set_xlabel('$x$ [m]')
    ax.set_ylabel('$y$ [m]')
    plt.show()
