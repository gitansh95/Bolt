#import arrayfire as af
import numpy as np
from scipy.signal import correlate
from scipy.optimize import curve_fit
#import glob
#import h5py
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
matplotlib.use('agg')
import pylab as pl


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


freq = np.arange(2.0, 3.1, 1.0)

# Plot
pl.ylabel('$\mathrm{\phi}$')
pl.xlabel('$\mathrm{y\ \mu m}$')

for index in freq:
    
    AC_freq        = index
    time_period    = 1/AC_freq
    #t_final        = params.t_final
    #transient_time = t_final/2.

    phase_corr = \
        np.loadtxt("data/phase_shift_corr_array_L_1.000_2.5000_tau_ee_inf_tau_eph_5.0_freq_%.1f.txt"%index)
    phase_fitting = \
        np.loadtxt("data/edge_density_L_1.000_2.5000_tau_ee_inf_tau_eph_5.0_freq_%.1f.txt"%index)
    q2  = np.loadtxt("data/q2_edge_L_1.000_2.500_tau_ee_inf_tau_eph_5.0_freq_%.1f.txt"%index)
    
    print ("phase_corr", phase_corr.shape)
    print ("phase_fitting", phase_fitting.shape)
    pl.plot(q2, phase_corr, '-o', label='$\mathrm{corr%d}$'%index)
    pl.plot(q2, phase_fitting, '-o', label='$\mathrm{fit%d}$'%index)
            
pl.title('$\mathrm{1.0 \\times 2.5,\ \\tau_{ee} = \infty,\ \\tau_{eph} = 5.0}$')
pl.legend(loc='best')
    
#pl.axvspan(sensor_1_left_start, sensor_1_left_end, color = 'k', alpha = 0.1)
#pl.axvspan(sensor_2_left_start, sensor_2_left_end, color = 'k', alpha = 0.1)    
    
pl.savefig('phase_vs_y.png')
pl.clf()

