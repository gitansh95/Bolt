import numpy as np
import h5py
import scipy.fftpack
import csv
import pylab as pl
import matplotlib.patches as patches

from scipy.optimize import curve_fit
from scipy.optimize import fsolve

%matplotlib inline

pl.rcParams['figure.figsize']  = 12, 7.5
pl.rcParams['lines.linewidth'] = 1.5
pl.rcParams['font.family']     = 'serif'
pl.rcParams['font.weight']     = 'bold'
pl.rcParams['font.size']       = 25

pl.rcParams['font.sans-serif'] = 'serif'
pl.rcParams['text.usetex']     = True
pl.rcParams['axes.linewidth']  = 1.5
pl.rcParams['axes.titlesize']  = 'medium'
pl.rcParams['axes.labelsize']  = 'medium'

pl.rcParams['xtick.major.size'] = 8
pl.rcParams['xtick.minor.size'] = 4
pl.rcParams['xtick.major.pad']  = 8
pl.rcParams['xtick.minor.pad']  = 8
pl.rcParams['xtick.color']      = 'k'
pl.rcParams['xtick.labelsize']  = 'medium'
pl.rcParams['xtick.direction']  = 'in'

pl.rcParams['ytick.major.size'] = 8
pl.rcParams['ytick.minor.size'] = 4
pl.rcParams['ytick.major.pad']  = 8
pl.rcParams['ytick.minor.pad']  = 8
pl.rcParams['ytick.color']      = 'k'
pl.rcParams['ytick.labelsize']  = 'medium'
pl.rcParams['ytick.direction']  = 'in'


tau = np.arange(0.01, 0.201, 0.01)
local_conductivity_array = []
nonlocal_conductivity_array = []

dump_step = 5
time_step = .025/4

q2_start = 0.0
q2_end   = 1.25
N_q2     = 90
q2       = q2_start + (0.5 + np.arange(N_q2)) * (q2_end - q2_start)/N_q2


local_indices = q2 < 0.25
nonlocal_indices = (q2 > 1.0) & (q2 < 1.25)

voltage_local_list    = []
voltage_nonlocal_list = []
time_list             = []

for index in tau:

    voltages    = np.load('tau_varying_L_1_1.25/voltages_L_1.0_2.5_tau_ee_inf_tau_eph_%.2f.txt.npz'%index)
    voltage_left = voltages['left']
    voltage_right = voltages['right']

    print (voltage_left.shape)
    print (voltage_right.shape)

    time = dump_step * time_step * np.arange(0, voltage_left[:, 0].size, 1)
    indices = np.where(time > time[-1]- 100)
    time_list.append(time[indices])

    left_probe = np.mean(voltage_left[:, local_indices], axis=1)
    right_probe = np.mean(voltage_right[:, local_indices], axis=1)
    voltage_local =  left_probe - right_probe

    left_probe = np.mean(voltage_left[:, nonlocal_indices], axis=1)
    right_probe = np.mean(voltage_right[:, nonlocal_indices], axis=1)
    voltage_nonlocal =  left_probe - right_probe

    #local_conductivity_array.append(1/voltage_local[-1])
    #nonlocal_conductivity_array.append(1/voltage_nonlocal[-1])

    #pl.plot(time[1000:], voltage_local[1000:], color='C0', alpha=1)
    #pl.plot(time[1000:], voltage_nonlocal[1000:], color='C1', alpha=1)
    #pl.plot(q2, voltage_left[-1, :], color='C0', alpha=1)
    #pl.plot(q2, voltage_right[-1, :], color='C0', alpha=1)
    #pl.savefig('tau_varying/images/current_L_1.0_%.1f.png'%l)
    #pl.clf()

    voltage_local_list.append(voltage_local[indices])
    voltage_nonlocal_list.append(voltage_nonlocal[indices])

for tau_index in range(2, tau.size):
    mean         = np.mean(voltage_local_list[tau_index])
    fluctuations = voltage_local_list[tau_index] - mean
    pl.semilogx(tau[tau_index] + np.zeros(voltage_local_list[tau_index].size), mean + 1e7*fluctuations, 'k.', alpha=0.1, markersize=0.3)
