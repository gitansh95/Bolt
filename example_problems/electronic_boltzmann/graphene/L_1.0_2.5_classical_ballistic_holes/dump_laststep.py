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
import yt
yt.enable_parallelism()

import petsc4py, sys; petsc4py.init(sys.argv)
from petsc4py import PETSc

from bolt.lib.physical_system import physical_system

from bolt.lib.nonlinear_solver.nonlinear_solver \
    import nonlinear_solver
from bolt.lib.nonlinear_solver.EM_fields_solver.electrostatic \
    import compute_electrostatic_fields

import domain
import boundary_conditions
import params
import initialize

import bolt.src.electronic_boltzmann.advection_terms as advection_terms

import bolt.src.electronic_boltzmann.collision_operator \
    as collision_operator

import bolt.src.electronic_boltzmann.moment_defs as moment_defs

# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 8, 8
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

source_start = params.contact_start
source_end   = params.contact_end

drain_start  = params.contact_start
drain_end    = params.contact_end

source_indices =  (q2 > source_start) & (q2 < source_end)
drain_indices  =  (q2 > drain_start)  & (q2 < drain_end )

# Left needs to be near source, right sensor near drain
sensor_1_left_start = 8.5 # um
sensor_1_left_end   = 9.5 # um

sensor_1_right_start = 8.5 # um
sensor_1_right_end   = 9.5 # um

sensor_1_left_indices  = (q2 > sensor_1_left_start ) & (q2 < sensor_1_left_end)
sensor_1_right_indices = (q2 > sensor_1_right_start) & (q2 < sensor_1_right_end)

sensor_2_left_start = 6.5 # um
sensor_2_left_end   = 7.5 # um

sensor_2_right_start = 6.5 # um
sensor_2_right_end   = 7.5 # um

sensor_2_left_indices  = (q2 > sensor_2_left_start ) & (q2 < sensor_2_left_end)
sensor_2_right_indices = (q2 > sensor_2_right_start) & (q2 < sensor_2_right_end)

filepath = \
'/home/mchandra/gitansh/gitansh_bolt/example_problems/electronic_boltzmann/L_5.0_10.0_classical_ballistic_angular_spread/dumps'
moment_files 		  = np.sort(glob.glob(filepath+'/moment*.h5'))
lagrange_multiplier_files = \
        np.sort(glob.glob(filepath+'/lagrange_multipliers*.h5'))

filepath_2 = \
'/home/mchandra/gitansh/gitansh_bolt/example_problems/electronic_boltzmann/L_5.0_10.0_classical_ballistic_angular_spread_holes/dumps'
moment_files_2 		  = np.sort(glob.glob(filepath_2+'/moment*.h5'))
lagrange_multiplier_files_2 = \
        np.sort(glob.glob(filepath_2+'/lagrange_multipliers*.h5'))

dt = params.dt
dump_interval = params.dump_steps

time_array = np.loadtxt("dump_time_array.txt")

for file_number, dump_file in yt.parallel_objects(enumerate(moment_files[::-1])):

    file_number = -1
    print("file number = ", file_number, "of ", moment_files.size)

    h5f  = h5py.File(dump_file, 'r')
    moments = np.swapaxes(h5f['moments'][:], 0, 1)
    h5f.close()


    density = moments[:, :, 0]
    j_x     = moments[:, :, 1]
    j_y     = moments[:, :, 2]

    h5f  = h5py.File(lagrange_multiplier_files[file_number], 'r')
    lagrange_multipliers = h5f['lagrange_multipliers'][:]
    h5f.close()

    mu    = lagrange_multipliers[:, :, 0]
    mu_ee = lagrange_multipliers[:, :, 1]
    T_ee  = lagrange_multipliers[:, :, 2]
    vel_drift_x = lagrange_multipliers[:, :, 3]
    vel_drift_y = lagrange_multipliers[:, :, 4]
    j_x_11 = lagrange_multipliers[:, :, 5]
    j_y_11 = lagrange_multipliers[:, :, 6]
    
    #vel_drift_x = np.divide(j_x, density).T
    #vel_drift_y = np.divide(j_y, density).T
    
    
    h5f  = h5py.File(moment_files_2[file_number], 'r')
    moments_2 = np.swapaxes(h5f['moments'][:], 0, 1)
    h5f.close()

    density_2 = moments_2[:, :, 0]
    j_x_2     = moments_2[:, :, 1]
    j_y_2     = moments_2[:, :, 2]

    h5f  = h5py.File(lagrange_multiplier_files_2[file_number], 'r')
    lagrange_multipliers_2 = h5f['lagrange_multipliers'][:]
    h5f.close()

    mu_2    = lagrange_multipliers_2[:, :, 0]
    mu_ee_2 = lagrange_multipliers_2[:, :, 1]
    T_ee_2  = lagrange_multipliers_2[:, :, 2]
    vel_drift_x_2 = lagrange_multipliers_2[:, :, 3]
    vel_drift_y_2 = lagrange_multipliers_2[:, :, 4]
    j_x_22 = lagrange_multipliers_2[:, :, 5]
    j_y_22 = lagrange_multipliers_2[:, :, 6]

    #vel_drift_x_2 = np.divide(j_x_2, density_2).T
    #vel_drift_y_2 = np.divide(j_y_2, density_2).T
  
#    print("file_number = ", file_number, "vel_drift_x.shape = ", vel_drift_x.shape)
#    print("file_number = ", file_number, "vel_drift_y.shape = ", vel_drift_y.shape)
   
    net_current_x = j_x_11 + j_x_22
    net_current_y = j_y_11 + j_y_22

    #pl.subplot(1,2,1)
    pl.contourf(q1_meshgrid, q2_meshgrid, density+density_2, 100, cmap='bwr')
    #pl.title(r'Time = ' + "%.2f"%(time_array[file_number]) + " ps")

    pl.streamplot(q1, q2, 
                  net_current_x, net_current_y,
                  density=4, color='k',
                  linewidth=1.0, arrowsize=1.7
                 )

    pl.gca().set_xticklabels([])
    pl.gca().set_yticklabels([])
    
    pl.xlim([q1[0], q1[-1]])
    pl.ylim([5., q2[-1]])
    
    pl.gca().set_aspect('equal')
    #pl.xlabel(r'$x\;(\mu \mathrm{m})$')
    #pl.ylabel(r'$y\;(\mu \mathrm{m})$')

    #pl.subplot(1,2,2)
    #pl.contourf(q1_meshgrid, q2_meshgrid, density+density_2, 100, cmap='bwr')
    #pl.title(r'Time = ' + "%.2f"%(time_array[file_number]) + " ps")

    #pl.xlim([domain.q1_start, domain.q1_end])
    #pl.ylim([domain.q2_start, domain.q2_end])
    
    #pl.colorbar()
    #pl.gca().set_aspect('equal')
    #pl.xlabel(r'$x\;(\mu \mathrm{m})$')
    #pl.ylabel(r'$y\;(\mu \mathrm{m})$')

    #pl.suptitle('$\\tau_\mathrm{mc} = \infty$ ps, $\\tau_\mathrm{mr} = 9.0$ ps')
    pl.savefig('images/dump_' + '%06d'%file_number + '.png', transparent = True)
    pl.clf()
    
