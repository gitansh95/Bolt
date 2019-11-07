import arrayfire as af
import numpy as np
from scipy.signal import correlate
from scipy.optimize import curve_fit
import glob
import h5py
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib import colors
matplotlib.use('agg')
import pylab as pl
import time
import yt
yt.enable_parallelism()

import petsc4py, sys; petsc4py.init(sys.argv)
from petsc4py import PETSc

import domain
#import boundary_conditions
#import params
#import initialize


# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 12, 7.5
pl.rcParams['figure.dpi']      = 100
pl.rcParams['image.cmap']      = 'jet'
pl.rcParams['lines.linewidth'] = 1.5
pl.rcParams['font.family']     = 'serif'
pl.rcParams['font.weight']     = 'normal'
pl.rcParams['font.size']       = 20
pl.rcParams['font.sans-serif'] = 'serif'
pl.rcParams['text.usetex']     = False
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

def sin_curve_fit(t, A, AC_freq, tau):
        return A*np.sin(2*np.pi*AC_freq*(t + tau ))

N_q1 = domain.N_q1
N_q2 = domain.N_q2

q1 = domain.q1_start + (0.5 + np.arange(N_q1)) * (domain.q1_end - domain.q1_start)/N_q1
q2 = domain.q2_start + (0.5 + np.arange(N_q2)) * (domain.q2_end - domain.q2_start)/N_q2

q2_meshgrid, q1_meshgrid = np.meshgrid(q2, q1)


dt = 0.025/2#params.dt
dump_interval = 5#params.dump_steps

freq_1 = np.arange(1.0, 49.91, 0.1)
frequencies = np.arange(50.0, 120.1, 1.0)
frequencies = np.append(freq_1, frequencies)

tolerance = 1.6
phase_arrays = np.loadtxt('phase_vs_freq_data/phase_nonlocal_vs_freq_L_1.000_2.500.txt')
# Fixing phase shifts by pi
flag = True
while (flag):
    flag = False
    for index in range(1, frequencies.size):
        #print ("Index ", index)
        if ((phase_arrays[index] - phase_arrays[index-1]) > tolerance):
            #print ("Freq ", AC_freq_array[index])
            phase_arrays[index] = phase_arrays[index] - np.pi
            flag = True

freq_in_GHz = frequencies * 10

time_shift_array = np.divide(phase_arrays, frequencies)/(2*np.pi/100.)

file_number = 0
for freq in frequencies[file_number:]:

    print("#############  Freq : ", freq)
    T = 100./freq #Time period
    filepath = \
    '/home/mchandra/gitansh/gitansh_bolt/example_problems/electronic_boltzmann/L_1.000_2.500_tau_ee_inf_tau_eph_5.0_freq_%.1f_rerun_5/dumps'%freq
    time_array                = np.loadtxt(filepath+"/../dump_time_array.txt")
    moment_files 		      = np.sort(glob.glob(filepath+'/moment*.h5'))
    lagrange_multiplier_files = \
        np.sort(glob.glob(filepath+'/lagrange_multipliers*.h5'))
 
    t0 = -1

    #pl.suptitle('$\\tau_\mathrm{mc} = \infty$ ps, $\\tau_\mathrm{mr} = 5.0$ ps')

    print ("Moment files : ", moment_files.size)

    h5f  = h5py.File(moment_files[t0], 'r')
    moments = np.swapaxes(h5f['moments'][:], 0, 1)
    h5f.close()

    density = moments[:, :, 0]
    j_x     = moments[:, :, 1]
    j_y     = moments[:, :, 2]

    delta_n = density - np.mean(density)
    
    h5f  = h5py.File(lagrange_multiplier_files[t0], 'r')
    lagrange_multipliers = h5f['lagrange_multipliers'][:]
    h5f.close()

    mu    = lagrange_multipliers[:, :, 0]
    mu_ee = lagrange_multipliers[:, :, 1]
    T_ee  = lagrange_multipliers[:, :, 2]
    vel_drift_x = lagrange_multipliers[:, :, 3]
    vel_drift_y = lagrange_multipliers[:, :, 4]
    
    N_cols = 5
    N_rows = 6
    gs = gridspec.GridSpec(N_rows, N_cols)
    ax1 = pl.subplot(gs[:4, 3:])
    ax2 = pl.subplot(gs[4:, :])
    ax3 = pl.subplot(gs[1:3, :3])

    print ("Time array.shape : ", time_array.shape)

    #ax1.set_title("Time = %.4f T, Freq = %.2f GHz"%(time_array[t0]/T, 1e3/T))
    ax1.contourf(q1_meshgrid, q2_meshgrid, delta_n, 200,
            cmap='bwr')
    ax1.streamplot(q1, q2, 
                  vel_drift_x, vel_drift_y,
                  density=1.8, color='k',
                  linewidth=0.9, arrowsize=1.2
                 )
    ax1.set_xlim([q1[0], q1[-1]])
    ax1.set_ylim([q2[0], q2[-1]])
    ax1.set_aspect('equal')
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    #ax1.set_xlabel(r'$x\;(\mu \mathrm{m})$')
    #ax1.set_ylabel(r'$y\;(\mu \mathrm{m})$')
    
    
    nonlocal_V = \
        np.loadtxt("edge_data/nonlocal_voltage_L_1.000_2.500_tau_ee_inf_tau_eph_5.0_freq_%.1f.txt"%freq)
    
    period = 100./freq
    time_indices = \
            (time_array > time_array[-1] - period)

    norm = np.max(np.abs(nonlocal_V[time_indices]))
    drive = np.sin(2*np.pi*freq*time_array[time_indices]/100.)

    normalized_signal = nonlocal_V[time_indices]/norm
    
    # Curve fitting measured signal
    initial_guess = [1, freq/100., 1]
    popt, pcov = curve_fit(sin_curve_fit,
                           time_array[time_indices],
                           normalized_signal,
                           p0 = initial_guess)

    #print ("Reconstruction A/f/tau : ", popt[0], popt[1], popt[2])


    # Reconstruction
    start_time = time_array[time_indices][-1] - 2*period
    end_time   = time_array[time_indices][-1]
    reconstruction_dt = 0.025/32
    reconstructed_time = np.arange(start_time, end_time, reconstruction_dt)

    reconstructed_signal = sin_curve_fit(reconstructed_time, popt[0], popt[1], popt[2])
    reconstructed_drive  = np.sin (2*np.pi*freq*reconstructed_time/100.)


    
    #ax3.plot(time_array[time_indices]/period,
    #         drive,
    #         color = 'k', ls = '-')

    #ax3.plot(time_array[time_indices]/period,
    #        nonlocal_V[time_indices]/norm,
    #        color = 'r', alpha = 0.5)

    ax3.plot(reconstructed_time/period,
             reconstructed_signal/np.max(np.abs(reconstructed_signal)),
             color='C1', ls='-', alpha = 1)

    ax3.plot(reconstructed_time/period,
             reconstructed_drive,
             color = 'k')

    ax3.axhline (0, color = 'k', ls = '--')
    
    ax3.set_xlim([time_array[-1]/period-2., time_array[-1]/period])
    ax3.set_ylim([-1.1, 1.1])

    #peak_position_signal = \
    #    (time_array[time_indices]/T)[np.argmin(nonlocal_V[time_indices]/norm)]
    #peak_position_drive  = (time_array[time_indices]/T)[np.argmin(drive)]
    
    
    #peak_position_signal = \
    #    (reconstructed_time/T)[np.argmax(reconstructed_signal)]
    peak_index_drive = np.argmax(reconstructed_drive[:int(reconstructed_time.size/2)])
    peak_position_drive  = (reconstructed_time/T)[peak_index_drive]

    time_shift = time_shift_array[file_number]/T

    #print ("tau : ", time_shift)
    peak_position_signal = peak_position_drive - time_shift

    if (phase_arrays[file_number] > 0):
        #peak_position_signal = \
        #        (reconstructed_time/T)[np.argmax(reconstructed_signal[:peak_index_drive])]
        
        ax3.axvspan(peak_position_signal,
                    peak_position_drive,
                    color = 'C0', alpha = 0.2)
    else :
        #peak_position_signal = \
        #        (reconstructed_time/T)[np.argmax(reconstructed_signal[peak_index_drive:])]
        ax3.axvspan(peak_position_drive,
                    peak_position_signal,
                    color = 'r', alpha = 0.2)
    print ("peaks : ", peak_position_drive, peak_position_signal)

    #if (peak_position_drive > peak_position_signal):
    #    ax3.axvspan(peak_position_signal,
    #                peak_position_drive, color = 'C0', alpha = 0.2)
    #else :
    #    ax3.axvspan(peak_position_drive, peak_position_signal, color = 'r', alpha = 0.2)
        
    ax3.set_xticks([])
    ax3.set_yticklabels([])


    ax2.plot(freq_in_GHz, phase_arrays, '-', lw = 2, color = 'C1')

    ax2.set_xlim([10, 1200])
    ax2.set_ylim([-10, 4])
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.axvline(freq_in_GHz[file_number], color = 'k', lw = 2)
    ax2.text(950, -3.2, "%.1f"%freq_in_GHz[file_number])
    ax2.axhline(0, ls = '--', color = 'k')
    ax2.plot(freq_in_GHz[file_number], phase_arrays[file_number], 'o', alpha = 0.9, color = 'r')

    pl.subplots_adjust(wspace = 2.5, hspace = 2.5)
    
    pl.savefig('images/dump_' + '%06d'%file_number + '.png')
    pl.clf()
    file_number += 1
