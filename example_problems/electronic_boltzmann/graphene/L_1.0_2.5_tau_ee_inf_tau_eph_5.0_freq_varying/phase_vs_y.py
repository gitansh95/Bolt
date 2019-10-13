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

import petsc4py, sys; petsc4py.init(sys.argv)
from petsc4py import PETSc

import domain
import boundary_conditions
import params
import initialize


# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 12, 7.5
pl.rcParams['figure.dpi']      = 100
pl.rcParams['image.cmap']      = 'jet'
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

def sin_curve_fit(t, A, tau):
        return A*np.sin(2*np.pi*AC_freq*(t + tau ))

N_q1 = domain.N_q1
N_q2 = domain.N_q2

q1 = domain.q1_start + (0.5 + np.arange(N_q1)) * (domain.q1_end - domain.q1_start)/N_q1
q2 = domain.q2_start + (0.5 + np.arange(N_q2)) * (domain.q2_end - domain.q2_start)/N_q2

q2_meshgrid, q1_meshgrid = np.meshgrid(q2, q1)

source_start = params.contact_start
source_end   = params.contact_end

drain_start  = params.contact_start
drain_end    = params.contact_end

source_indices =  (q2 > source_start) & (q2 < source_end)
drain_indices  =  (q2 > drain_start)  & (q2 < drain_end )

# Left needs to be near source, right sensor near drain
sensor_1_left_start = source_end # um
sensor_1_left_end   = domain.q2_end # um

sensor_1_right_start = drain_end # um
sensor_1_right_end   = domain.q2_end # um

sensor_1_left_indices  = (q2 > sensor_1_left_start ) & (q2 < sensor_1_left_end)
sensor_1_right_indices = (q2 > sensor_1_right_start) & (q2 < sensor_1_right_end)

freq = np.arange(2.0, 50.1, 1.0)

for index in freq:
    filepath = \
    '/home/mchandra/gitansh/gitansh_bolt/example_problems/electronic_boltzmann/L_1.000_2.500_tau_ee_inf_tau_eph_5.0_freq_%.1f/dumps'%index
    moment_files 		  = np.sort(glob.glob(filepath+'/moment*.h5'))
    
    AC_freq        = index
    time_period    = 1/AC_freq
    t_final        = params.t_final
    transient_time = t_final/2.

    time         = np.loadtxt("data/time_L_1.000_2.500_tau_ee_inf_tau_eph_5.0_freq_%.1f.txt"%index)
    edge_density = np.loadtxt("data/edge_density_L_1.000_2.5000_tau_ee_inf_tau_eph_5.0_freq_%.1f.txt"%index)
    q2           = np.loadtxt("data/q2_edge_L_1.000_2.500_tau_ee_inf_tau_eph_5.0_freq_%.1f.txt"%index)
    
    N_spatial = edge_density.shape[1]
    
    transient_index = int((transient_time/t_final)*time.size)
        
    drive = np.sin(2*np.pi*AC_freq*time)
    nsamples = drive.size
    dt_corr = np.linspace(-time[-1] + time[0],\
            time[-1] - time[0], 2*nsamples-1)
    
    # Discarding transients
    q = q2.size/2
    time_half = time[transient_index:]
    drive_half = drive[transient_index:]
    
    # Plotting signals at edge
    norm_0 = np.max(edge_density[transient_index:, 0])
    norm_1 = np.max(edge_density[transient_index:, -1])
    
    pl.plot(time, drive, color='black', linestyle='--')
    pl.ylim([-1.1, 1.1])
    pl.xlim([0, t_final])
    pl.xlabel('$\mathrm{Time\;(s)}$')
    
    for i in range(N_spatial):
        norm_i = np.max(edge_density[transient_index:, i])
        pl.plot(time, edge_density[:, i]/norm_i)
    
    pl.savefig('images/signals.png')
    pl.clf()
    
    phase_shift_corr_array = []
    phase_shift_fitting_array = []\
    
    for i in range(N_spatial):
        print ('index : ', i)
        signal_1 = edge_density[:, i]
        norm_1 = np.max(signal_1[transient_index:])
        signal_1_normalized = signal_1/norm_1
            
        # Calculate phase_shifts using scipy.correlate
        corr = correlate(drive, signal_1_normalized)
        time_shift_corr = dt_corr[corr.argmax()]
        phase_shift_corr  = 2*np.pi*(((0.5 + time_shift_corr/time_period) % 1.0) - 0.5)
    
        # Calculate phase_shifts using scipy.curve_fit
        popt, pcov = curve_fit(sin_curve_fit, time[transient_index:],\
                    signal_1_normalized[transient_index:])
        time_shift_fitting = popt[1]%(time_period/2.0)
        phase_shift_fitting  = 2*np.pi*(((0.5 + time_shift_fitting/time_period) % 1.0) - 0.5)
    
        phase_shift_corr_array.append(phase_shift_corr)
        phase_shift_fitting_array.append(phase_shift_fitting)
    
    phase_shift_corr_array = np.array(phase_shift_corr_array)
    phase_shift_fitting_array = np.array(phase_shift_fitting_array)


    np.savetxt("data/phase_shift_corr_array_L_1.000_2.5000_tau_ee_inf_tau_eph_5.0_freq_%.1f.txt"%index, phase_shift_corr_array)
    np.savetxt("data/phase_shift_fitting_array_L_1.000_2.5000_tau_ee_inf_tau_eph_5.0_freq_%.1f.txt"%index, phase_shift_fitting_array)
    #np.savetxt('q2.txt', q2)

    pl.plot(q2, phase_shift_corr_array, '-o', label='$\mathrm{corr%.1f}$'%index)
    #pl.plot(q2, phase_shift_fitting_array, '-o', label='$\mathrm{fit}$')
    
# Plot
pl.ylabel('$\mathrm{\phi}$')
pl.xlabel('$\mathrm{y\ \mu m}$')
        
pl.title('$\mathrm{2.5 \\times 10,\ \\tau_{ee} = \infty,\ \\tau_{eph} = 2.5}$')
pl.legend(loc='best')
    
#pl.axvspan(sensor_1_left_start, sensor_1_left_end, color = 'k', alpha = 0.1)
#pl.axvspan(sensor_2_left_start, sensor_2_left_end, color = 'k', alpha = 0.1)    
    
pl.savefig('images/phase_vs_y.png')
pl.clf()

