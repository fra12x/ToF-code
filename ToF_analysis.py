# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 14:35:13 2023

@author: Francesco Mirani
"""

import numpy as np 
from math import pi
import math
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from mpl_point_clicker import clicker
from mpl_interactions import zoom_factory, panhandler
from typing import Tuple
from scipy.optimize import curve_fit


# Fubction to get the index of the closest value in array
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# Functions to add and remove points from figure
def point_added_cb(position: Tuple[float, float], klass: str):
    x, y = position

def point_removed_cb(position: Tuple[float, float], klass: str, idx):
    x, y = position
    suffix = {'1': 'st', '2': 'nd', '3': 'rd'}.get(str(idx)[-1], 'th')
    print(f"The {idx}{suffix} point of class {klass} with position {x=:.2f}, {y=:.2f}  was removed")
    
def get_rsq(func, x, y, popt):
    ss_res = np.dot((y - func(x, *popt)),(y - func(x, *popt)))
    ymean = np.mean(y)
    ss_tot = np.dot((y-ymean),(y-ymean))
    return 1-ss_res/ss_tot


################
# Get the data #
################
# H.p. only protons 
# time assumed in Seconds
Name_folder_and_file = "Data/SHOT 19.txt"
time, signal = np.genfromtxt(Name_folder_and_file, delimiter = ',', comments='#', usecols= (0,1), skip_header=21, unpack=True)
signal = -signal # if negative signals are provided
time_end_signal = 1e-6 # time at which the signal is over (for first bg calculation)
Source_det_dist = 1.83 #Distance from source [m]
thickness_detector = 100e-4 #cm
# Fit parameters for the signal
Temp = 0.5
vel_speces = 1.0e4
Amp_signal = 1e-36
# Get ranges of protons in Si (from https://physics.nist.gov/PhysRefData/Star/Text/PSTAR.html)
file_name = "Mass_Range_Si.txt"
E_Si, R_Si = np.genfromtxt(file_name, delimiter = ' ', comments='#', usecols= (0,1), skip_header=7, unpack=True)
density_Si = 2.329 # g/cm3
# Get renges of protons in Al (from https://physics.nist.gov/PhysRefData/Star/Text/PSTAR.html)
file_name = "Mass_Range_Al.txt"
E_Al, R_Al = np.genfromtxt(file_name, delimiter = ' ', comments='#', usecols= (0,1), skip_header=7, unpack=True)
density_Al = 2.7 # g/cm3
# Costants
prot_mass = 1.67*10**(-27) # [kg]
prot_mass_MeV = 938.272 #Proton mass [MeV/c2]
k_boltz = 1.38*10**(-23) # [m2 kg s-2 K-1]
c = 299792458 #m/s speed of light


################
# Process data #
################
# Remuve nans and inf values if present
time = time[np.isfinite(signal)]
signal = signal[np.isfinite(signal)]
# Get position of photopeak
id_fotopeak = find_nearest(abs(signal), max(abs(signal)))
start_time = time[id_fotopeak]
# Traslate signal in time
time = time - start_time
time_end_signal = time_end_signal - start_time
# Get the interval of iterest to consider the data
id_end_signal = find_nearest(time, time_end_signal)
mean_bg_out_interest = np.mean(signal[id_end_signal:])
std_bg_out_interest = np.std(signal[id_end_signal:])
mean_signal = mean_bg_out_interest
i = 0
while mean_signal < (mean_bg_out_interest + std_bg_out_interest):
    id_end_signal = id_end_signal - 100*i
    mean_signal = np.mean(signal[id_end_signal - 100 : id_end_signal])
    i = i + 1
time = time[id_fotopeak:id_end_signal + round((id_end_signal - id_fotopeak)/3)]*1e9 # ns time
signal = signal[id_fotopeak:id_end_signal + round((id_end_signal - id_fotopeak)/3)]
# plot the signal in range of interest
fig, (ax1, ax2) = plt.subplots(1,2)
fig.set_figheight(5)
fig.set_figwidth(10)
ax1.set_xlabel("Time  [ns]")
ax1.set_ylabel("Signal")
ax2.set_xlabel("Energy  [MeV]")
ax2.set_ylabel("Spectrum [arb. units]")
ax1.set_xlim([time[0], time[-1]])
ax1.plot([time[0], time[-1]], [mean_bg_out_interest, mean_bg_out_interest], color ="black", linewidth = 0.5) 
ax1.scatter(time, signal, color ="crimson", s = 10, edgecolors = "black", linewidths = 0.5) 
fig.tight_layout(w_pad=8.0)
# add zooming and middle click to pan
zoom_factory(ax1)
ph = panhandler(fig, button=2)
# Get window of data interactively
klicker = clicker(
   ax1,
   ["Edges"],
   markers=["x"]
)
klicker.on_point_added(point_added_cb)
klicker.on_point_removed(point_removed_cb)
# Get the initial point for the fit
while True:
    peak_init_pos = klicker.get_positions().get('Edges')
    plt.pause(0.01)
    if len(peak_init_pos) > 0:
        break
time_fit = time[find_nearest(time, peak_init_pos[0,0]):] + Source_det_dist/c*1e9
signal_fit = signal[find_nearest(time, peak_init_pos[0,0]):]
################
# Fit the data #
################
# Function for the fit
def func_signal(t, T, v_s, F_0):
    T = T*1.16e10 # MeV to K
    t = t*1e-9 #ns to s
    m = prot_mass # [kg]
    L = Source_det_dist #Distance from source [m]
    k = k_boltz # [m2 kg s-2 K-1]
    return mean_bg_out_interest + (L**2)/(t**5)*F_0*np.exp((-1*m/(2*k*T))*((L/t)-v_s)**2)
# Initial fit parameters
x0 = [Temp, vel_speces, Amp_signal]    
popt, pcov = curve_fit(func_signal, time_fit, signal_fit, p0 = x0, method = 'lm')
# Print R2
print("R-sqare fit: " + str(get_rsq(func_signal, time_fit, signal_fit, popt)))
# Get funtion and plot
time_inter = np.linspace(time_fit[0], time_fit[-1],1000)
signal_inter = func_signal(time_inter,*popt)
ax1.plot(time_inter, signal_inter,'b-', alpha = 1, lw = 2,label='soft_l1 loss')
# Eval maximum of the gradient and find intercept
grad_signal = np.diff(signal_inter)/(time_inter[1] - time_inter[0])
m_signal = max(grad_signal)
x_signal = time_inter[find_nearest(grad_signal, m_signal)]
y_signal = signal_inter[find_nearest(grad_signal, m_signal)]
y_base = mean_bg_out_interest
x_base = x_signal - y_signal/m_signal + y_base/m_signal
x_base = x_base 
# plot intercept
ax1.plot([x_base, x_signal], [y_base, y_signal], color ="black", linewidth = 0.5) 
#####################
# Generate spectrum #
#####################
# Get data from mimimum to maximum time
signal_fit = signal_fit[find_nearest(time_fit, x_base):]
time_fit = time_fit[find_nearest(time_fit, x_base):]
# Eval corresponding energies for proton
beta = np.array(Source_det_dist/(c*time_fit*1e-9)) # time in s
Energy_fit = prot_mass_MeV*((1.0/np.sqrt(1.0-beta**2))-1.0)
# Eval energy released in the detector 
E_Si_interp = np.linspace(0, max(Energy_fit),1000)
R_Si_interp = np.interp(E_Si_interp, E_Si, R_Si)
Energy_fit_deposit = np.zeros(len(Energy_fit))
Energy_fit_fraction = np.zeros(len(Energy_fit))
R_out = np.zeros(len(Energy_fit))
for i in np.arange(len(Energy_fit)):  
    id_ener_in = find_nearest(E_Si_interp, Energy_fit[i])
    R_out[i] = R_Si_interp[id_ener_in] - thickness_detector*density_Si
    if R_out[i] > 0:
        id_R_out = find_nearest(R_Si_interp, R_out[i])
        Energy_fit_deposit[i] = Energy_fit[i] - E_Si_interp[id_R_out]
    else:
        Energy_fit_deposit[i] = Energy_fit[i]
    Energy_fit_fraction[i] =  Energy_fit_deposit[i]/Energy_fit[i]

# Eval spectrum in arbitrary units
delta_time_fit = time_fit[1] - time_fit[0]
dN_dE = (signal_fit*Energy_fit_fraction)/(Energy_fit**2)*(0.5*time_fit + delta_time_fit)
        
ax2.plot(Energy_fit, dN_dE, linewidth = 1.5, drawstyle = "steps-mid", color = "steelblue")
ax2.fill_between(Energy_fit, dN_dE, step = "mid", color = "steelblue", alpha = 0.2)
ax2.set_yscale("log")
#######################
# Save image and data #
#######################
fig.savefig(Name_folder_and_file[:-4] + ' results.png')
np.savetxt(Name_folder_and_file[:-4] + ' results_prot.txt', np.column_stack([Energy_fit, dN_dE]))
