import arrayfire as af
import numpy as np
from scipy.signal import correlate
import glob
import h5py
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
matplotlib.use('agg')
import pylab as pl
import time
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

N_q1 = domain.N_q1
N_q2 = domain.N_q2

q1 = domain.q1_start + (0.5 + np.arange(N_q1)) * (domain.q1_end - domain.q1_start)/N_q1
q2 = domain.q2_start + (0.5 + np.arange(N_q2)) * (domain.q2_end - domain.q2_start)/N_q2

q2_meshgrid, q1_meshgrid = np.meshgrid(q2, q1)


dt = params.dt
dump_interval = params.dump_steps


freq_1 = np.arange(1.0, 49.91, 0.1)
frequencies = np.arange(50.0, 100.1, 1.0)
frequencies = np.append(freq_1, frequencies)
file_number = 0

for freq in frequencies:

    print("freq : ", freq)
    T = 100./freq #Time period
    filepath = \
    '/home/mchandra/gitansh/gitansh_bolt/example_problems/electronic_boltzmann/L_1.000_2.500_tau_ee_inf_tau_eph_5.0_freq_%.1f_rerun_3/dumps'%freq
    time_array                = np.loadtxt(filepath+"/../dump_time_array.txt")
    moment_files 		      = np.sort(glob.glob(filepath+'/moment*.h5'))
    lagrange_multiplier_files = \
        np.sort(glob.glob(filepath+'/lagrange_multipliers*.h5'))
 
    #val = 2.25*T # Look for index of timestamp 2T in the time array
    #tol = 2.5*dt # Float closeness tolerance
    #t0 = (np.abs(time_array - val) <= tol).argmax()
    #print ('t0 : ', t0, time_array[t0])

    t0 = -1

    pl.suptitle('$\\tau_\mathrm{mc} = \infty$ ps, $\\tau_\mathrm{mr} = 5.0$ ps')

    print ("Moment files : ", moment_files.size)

    h5f  = h5py.File(moment_files[t0], 'r')
    moments = np.swapaxes(h5f['moments'][:], 0, 1)
    h5f.close()

    density = moments[:, :, 0]
    j_x     = moments[:, :, 1]
    j_y     = moments[:, :, 2]
    
    h5f  = h5py.File(lagrange_multiplier_files[t0], 'r')
    lagrange_multipliers = h5f['lagrange_multipliers'][:]
    h5f.close()

    mu    = lagrange_multipliers[:, :, 0]
    mu_ee = lagrange_multipliers[:, :, 1]
    T_ee  = lagrange_multipliers[:, :, 2]
    vel_drift_x = lagrange_multipliers[:, :, 3]
    vel_drift_y = lagrange_multipliers[:, :, 4]
    
    ax1 = pl.gca()

    print ("Time array.shape : ", time_array.shape)

    ax1.set_title("Time = %.4f T, Freq = %.2f GHz"%(time_array[t0]/T, 1e3/T))
    ax1.contourf(q1_meshgrid, q2_meshgrid, density, 100, cmap='bwr')
    ax1.streamplot(q1, q2, 
                  vel_drift_x, vel_drift_y,
                  density=2, color='blue',
                  linewidth=0.7, arrowsize=1
                 )
    ax1.set_xlim([domain.q1_start, domain.q1_end])
    ax1.set_ylim([domain.q2_start, domain.q2_end])
    ax1.set_aspect('equal')
    #ax1.set_xlabel(r'$x\;(\mu \mathrm{m})$')
    ax1.set_ylabel(r'$y\;(\mu \mathrm{m})$')
    
    pl.savefig('images/dump_' + '%06d'%file_number + '.png')
    pl.clf()
    file_number += 1
