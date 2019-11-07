import scipy.fftpack
import csv
from scipy.optimize import root
from scipy.interpolate import interp1d

import arrayfire as af
import numpy as np
from scipy.signal import correlate
from scipy.optimize import curve_fit
import glob
import h5py
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
matplotlib.use('agg')
import pylab as pl
import yt
yt.enable_parallelism()

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

def line (x, m, c):
    return (m*x + c)

def sin_curve_fit(t, A, tau_in):
        return A*np.sin(2*np.pi*AC_freq*(t + tau_in ))

lengths = np.arange(1.0, 1.01, 0.25)

freq_1 = np.arange(1.0, 49.91, 0.1)
freq = np.arange(50.0, 120.1, 1.0)
freq = np.append(freq_1, freq)

AC_freq_array = freq/100.0

l = 1.0

L1 = l
L2 = 2.5*l

phase_vs_freq_array_1 = []
phase_vs_freq_array_2 = []
for index in freq:

    AC_freq = index/100.0
    time_period = 1/AC_freq

    time       = np.loadtxt("edge_data/time_L_%.3f_%.3f_tau_ee_inf_tau_eph_5.0_freq_%.1f.txt"%(L1, L2, index))
    q2         = np.loadtxt("edge_data/q2_edge_L_%.3f_%.3f_tau_ee_inf_tau_eph_5.0_freq_%.1f.txt"%(L1, L2, index))
    nonlocal_V = np.loadtxt("edge_data/nonlocal_voltage_L_%.3f_%.3f_tau_ee_inf_tau_eph_5.0_freq_%.1f.txt"%(L1, L2, index))

    half_time = int(time.shape[0]/2)

    drive = np.sin(2*np.pi*AC_freq*time)
    nsamples = drive.size
    dt_corr = np.linspace(-time[-1] + time[0],\
            time[-1] - time[0], 2*nsamples-1)

    # Discarding transients
    q = q2.size/2
    time_half = time[half_time:]
    drive_half = drive[half_time:]

    #print ('index : ', i)
    signal_1 = nonlocal_V
    norm_1 = np.max(signal_1[half_time:])
    signal_1_normalized = signal_1/norm_1

    # Calculate phase_shifts using scipy.correlate
    corr = correlate(drive, signal_1_normalized)
    time_shift_corr = dt_corr[corr.argmax()]
    phase_shift_corr  = 2*np.pi*(((0.5 + time_shift_corr/time_period) % 1.0) - 0.5)

    # Calculate phase_shifts using scipy.curve_fit
    initial_guess = [1, 1]
    popt, pcov = curve_fit(sin_curve_fit, time[half_time:],\
                signal_1_normalized[half_time:])
    time_shift_fitting = popt[1]%(time_period/2.0)
    phase_shift_fitting  = 2*np.pi*(((0.5 + time_shift_fitting/time_period) % 1.0) - 0.5)


    tolerance = 1.6
    tolerance_2 = np.pi

    # Match fitting method with corr method
    if (phase_shift_fitting - phase_shift_corr > tolerance):
        phase_shift_fitting = phase_shift_fitting - np.pi
    if (phase_shift_corr - phase_shift_fitting > tolerance):
        phase_shift_fitting = phase_shift_fitting + np.pi


    phase_vs_freq_array_1.append(phase_shift_corr)
    phase_vs_freq_array_2.append(phase_shift_fitting)


phase_vs_freq_array_1 = np.array(phase_vs_freq_array_1)
phase_vs_freq_array_2 = np.array(phase_vs_freq_array_2)


## Do not constrain phase to remain within -pi and pi
for f in range(freq.size-1):
    if ((phase_vs_freq_array_2[f] - phase_vs_freq_array_2[f+1]) > tolerance_2):
        phase_vs_freq_array_2[f+1] = phase_vs_freq_array_2[f+1] + 2*np.pi
    if ((phase_vs_freq_array_2[f+1] - phase_vs_freq_array_2[f]) > tolerance_2):
        phase_vs_freq_array_2[f+1] = phase_vs_freq_array_2[f+1] - 2*np.pi


flag = True
while (flag):
    flag = False
    for index in range(1, AC_freq_array.size):
        #print ("Index ", index)
        if ((phase_vs_freq_array_2[index] - phase_vs_freq_array_2[index-1]) > tolerance):
            phase_vs_freq_array_2[index] = phase_vs_freq_array_2[index] - np.pi
            flag = True

phase_vs_freq_array_2 = phase_vs_freq_array_2 + 2*np.pi

pl.plot(AC_freq_array, phase_vs_freq_array_1, '-o', label = "Corr", alpha = 0.5)
pl.plot(AC_freq_array, phase_vs_freq_array_2, '-o', label = "Fit", alpha = 0.5)
np.savetxt('phase_vs_freq_data/phase_nonlocal_vs_freq_L_1.000_2.500.txt',
        phase_vs_freq_array_2)
#pl.plot(AC_freq_array, phase_vs_freq_array_2, '-o', label = "$\mathrm{Top\ Edge}$", color = 'cyan', alpha = 0.5)

pl.axhline(0, color = 'k', ls = '--')

# Plot
pl.ylabel('$\mathrm{\phi}$')
pl.xlabel('$\mathrm{f\ (ps^{-1})}$')

pl.legend(loc='best', prop={'size':10})
pl.savefig('images/phase_vs_freq.png')
pl.clf()
