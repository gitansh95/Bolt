import arrayfire as af
import numpy as np
from scipy.signal import correlate
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
N_q2 = 30#domain.N_q2
q2_end = 1.25

q1 = domain.q1_start + (0.5 + np.arange(N_q1)) * (domain.q1_end - domain.q1_start)/N_q1
q2 = domain.q2_start + (0.5 + np.arange(N_q2)) * (q2_end - domain.q2_start)/N_q2

q2_meshgrid, q1_meshgrid = np.meshgrid(q2, q1)

T_1 = 100.
T_2 = 1.

dt = params.dt
dump_interval = params.dump_steps

filepath = \
'/home/mchandra/gitansh/gitansh_bolt/example_problems/electronic_boltzmann/L_1.000_2.500_tau_ee_inf_tau_eph_5.0_freq_1.0_rerun_4/dumps'
time_array                = \
    np.loadtxt("../L_1.000_2.500_tau_ee_inf_tau_eph_5.0_freq_1.0_rerun_4/dump_time_array.txt")
moment_files 		      = np.sort(glob.glob(filepath+'/moment*.h5'))
lagrange_multiplier_files = \
        np.sort(glob.glob(filepath+'/lagrange_multipliers*.h5'))

filepath_2 = \
    '/home/mchandra/gitansh/gitansh_bolt/example_problems/electronic_boltzmann/L_1.000_2.500_tau_ee_inf_tau_eph_5.0_freq_100.0_rerun_6/dumps'
time_array_2                  = np.loadtxt(filepath_2+"/../dump_time_array.txt")
moment_files_2                = np.sort(glob.glob(filepath_2+'/moment*.h5'))
lagrange_multiplier_files_2   = \
        np.sort(glob.glob(filepath_2+'/lagrange_multipliers*.h5'))

transient_index = int(time_array.size/3)
add_index = moment_files.size - transient_index
print ('moment : ', moment_files.size)
print ('moment2 : ', moment_files_2[-add_index:].size)

half_time_1 = int(time_array.size/2)
half_time_2 = int(3*time_array_2.size/4)

nonlocal_V_1 = \
    np.loadtxt('data/nonlocal_voltage_L_1.000_2.500_tau_ee_inf_tau_eph_5.0_freq_1.0.txt')
nonlocal_V_2 = \
    np.loadtxt('data/nonlocal_voltage_L_1.000_2.500_tau_ee_inf_tau_eph_5.0_freq_100.0.txt')

norm_1 = np.max(np.abs(nonlocal_V_1[half_time_1:]))
norm_2 = np.max(np.abs(nonlocal_V_2[half_time_2:]))

#######

fig = pl.figure()            

N_cols = 31
N_rows = 6
gs = gridspec.GridSpec(N_rows, N_cols)
ax1 = pl.subplot(gs[:4, :15])
ax2 = pl.subplot(gs[:4, 16:])
ax3 = pl.subplot(gs[4:, :])
#cax = pl.subplot (gs[:4, 15])

file_number = 3050
#for file_number, dump_file in yt.parallel_objects(enumerate(moment_files)):
while (file_number < 3201):

    print("file number = ", file_number, "of ", moment_files.size)

    h5f  = h5py.File(moment_files[file_number+transient_index], 'r')
    moments = np.swapaxes(h5f['moments'][:], 0, 1)
    h5f.close()
    print (moment_files[file_number+transient_index])

    density = moments[:, :, 0]
    j_x     = moments[:, :, 1]
    j_y     = moments[:, :, 2]
    
    delta_n = density - np.mean(density)
    
    
    h5f  = h5py.File(moment_files_2[file_number-add_index], 'r')
    moments_2 = np.swapaxes(h5f['moments'][:], 0, 1)
    h5f.close()
    print (moment_files_2[file_number-add_index])
    
    density_2 = moments_2[:, :, 0]
    j_x_2     = moments_2[:, :, 1]
    j_y_2     = moments_2[:, :, 2]

    delta_n_2 = density_2 - np.mean(density_2)


    h5f  = h5py.File(lagrange_multiplier_files[file_number+transient_index], 'r')
    lagrange_multipliers = h5f['lagrange_multipliers'][:]
    h5f.close()

    mu    = lagrange_multipliers[:, :, 0]
    mu_ee = lagrange_multipliers[:, :, 1]
    T_ee  = lagrange_multipliers[:, :, 2]
    vel_drift_x = lagrange_multipliers[:, :, 3]
    vel_drift_y = lagrange_multipliers[:, :, 4]
    
    
    h5f  = h5py.File(lagrange_multiplier_files_2[file_number-add_index], 'r')
    lagrange_multipliers_2 = h5f['lagrange_multipliers'][:]
    h5f.close()
    
    mu_2    = lagrange_multipliers_2[:, :, 0]
    mu_ee_2 = lagrange_multipliers_2[:, :, 1]
    T_ee_2  = lagrange_multipliers_2[:, :, 2]
    vel_drift_x_2 = lagrange_multipliers_2[:, :, 3]
    vel_drift_y_2 = lagrange_multipliers_2[:, :, 4]
  
     
    #ax1.set_title("Time = %.4f T"%(time_array[file_number]/T_1))
    ax1.contourf(q1_meshgrid, q2_meshgrid, delta_n, 200,
            cmap='bwr')
    ax1.streamplot(q1, q2, 
                  vel_drift_x, vel_drift_y,
                  density=2, color='k',
                  linewidth=0.7, arrowsize=1
                 )
    ax1.set_xlim([q1[0], q1[-1]])
    ax1.set_ylim([q2[0], q2[-1]])
    ax1.set_aspect('equal')
    #ax1.set_xlabel(r'$x\;(\mu \mathrm{m})$')
    #ax1.set_ylabel(r'$y\;(\mu \mathrm{m})$')
    ax1.set_xticks([])
    ax1.set_yticks([])
    

    #ax2.set_title("Time =  %.4f T"%(time_array_2[file_number-add_index]/T_2-6.0))
    ax2.contourf(q1_meshgrid, q2_meshgrid, delta_n_2, 200,
            cmap='bwr')
    ax2.streamplot(q1, q2, 
                  vel_drift_x_2, vel_drift_y_2,
                  density=2, color='k',
                  linewidth=0.7, arrowsize=1
                 )
    ax2.set_xlim([q1[0], q1[-1]])
    ax2.set_ylim([q2[0], q2[-1]])     
    ax2.set_aspect('equal')
    #ax3.set_xlabel(r'$x\;(\mu \mathrm{m})$')
    #ax2.set_ylabel(r'$y\;(\mu \mathrm{m})$')
    ax2.set_yticks([])
    ax2.set_xticks([])
    
    
    ax3.set_title("%.2f"%(time_array[file_number]/T_1), pad=16.)
    ax3.set_xlim([1.0, 3.0])
    ax3.set_ylim([-1.1, 1.1])
    ax3.plot(time_array[transient_index:]/T_1,
            np.sin(2*np.pi*0.01*time_array[transient_index:]),
            color='k',
            alpha=1, ls = '-')
    ax3.plot(time_array[transient_index:]/T_1, nonlocal_V_1[transient_index:]/norm_1, color = 'C0')
    ax3.axvline(time_array[file_number+transient_index]/T_1, color = 'k')
    ax3.axhline(0, color = 'k', ls = '--')
    #ax3.set_xlabel('$T_0$')

    #ax3.plot(time_array_2[-add_index:]/T_2-6.0,
    #        np.sin(2*np.pi*time_array_2)[-add_index:], color='r', alpha=0.5, ls = '--')
    ax3.plot(time_array_2[-add_index:]/T_2-22.0, nonlocal_V_2[-add_index:]/norm_2, color = 'C1')

    ax3.set_xticks([1.5, 2.0, 2.5])
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])

    pl.subplots_adjust(wspace = 2.5, hspace = 2.5)

    pl.savefig('images/dump_dual_' + '%06d'%(file_number) + '.png')
    file_number +=1
    
    ax1.clear()
    ax2.clear()
    ax3.clear()
