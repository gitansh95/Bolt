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

import domain_1
import boundary_conditions_1
import params_1
import initialize_1

import domain_2
import boundary_conditions_2
import params_2
import initialize_2

import domain_3
import boundary_conditions_3
import params_3
import initialize_3

import bolt.src.electronic_boltzmann.advection_terms as advection_terms

import bolt.src.electronic_boltzmann.collision_operator \
    as collision_operator

import bolt.src.electronic_boltzmann.moment_defs as moment_defs

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

filepath = \
'/home/mchandra/gitansh/zero_T_with_mirror/example_problems/electronic_boltzmann/graphene/cross/dumps'
#'/root/bolt/Bolt/example_problems/electronic_boltzmann/graphene/L_1.0_1.0_tau_ee_inf_tau_eph_1000.0_DC_L_bent/dumps'

N_q1_1 = domain_1.N_q1
N_q2_1 = domain_1.N_q2

q1_1 = domain_1.q1_start + (0.5 + np.arange(N_q1_1)) * (domain_1.q1_end - \
        domain_1.q1_start)/N_q1_1
q2_1 = domain_1.q2_start + (0.5 + np.arange(N_q2_1)) * (domain_1.q2_end - \
        domain_1.q2_start)/N_q2_1

dq1_1 = (domain_1.q1_end - domain_1.q1_start)/N_q1_1
dq2_1 = (domain_1.q2_end - domain_1.q2_start)/N_q2_1

q2_meshgrid_1, q1_meshgrid_1 = np.meshgrid(q2_1, q1_1)


N_q1_2 = domain_2.N_q1
N_q2_2 = domain_2.N_q2

q1_2 = domain_2.q1_start + (0.5 + np.arange(N_q1_2)) * (domain_2.q1_end - \
        domain_2.q1_start)/N_q1_2
q2_2 = domain_2.q2_start + (0.5 + np.arange(N_q2_2)) * (domain_2.q2_end - \
        domain_2.q2_start)/N_q2_2

dq1_2 = (domain_2.q1_end - domain_1.q1_start)/N_q1_1
dq2_2 = (domain_2.q2_end - domain_1.q2_start)/N_q2_1

q2_meshgrid_2, q1_meshgrid_2 = np.meshgrid(q2_2, q1_2)


N_q1_3 = domain_3.N_q1
N_q2_3 = domain_3.N_q2

q1_3 = domain_3.q1_start + (0.5 + np.arange(N_q1_3)) * (domain_3.q1_end - \
        domain_3.q1_start)/N_q1_3
q2_3 = domain_3.q2_start + (0.5 + np.arange(N_q2_3)) * (domain_3.q2_end - \
        domain_3.q2_start)/N_q2_3

dq1_3 = (domain_3.q1_end - domain_3.q1_start)/N_q1_3
dq2_3 = (domain_3.q2_end - domain_3.q2_start)/N_q2_3

q2_meshgrid_3, q1_meshgrid_3 = np.meshgrid(q2_3, q1_3)


N_q1 = N_q1_1 + N_q1_2 + N_q1_3
N_q2 = N_q2_2

q1_start = 0.
q1_end   = domain_1.q1_end + domain_2.q1_end + domain_3.q1_end
q2_start = 0.
q2_end   = domain_2.q2_end

dq1 = (q1_end - q1_start)/N_q1
dq2 = (q2_end - q2_start)/N_q2

q1 = 0. + (0.5 + np.arange(N_q1)) * dq1
q2 = 0. + (0.5 + np.arange(N_q2)) * dq2

q2_meshgrid, q1_meshgrid = np.meshgrid(q2, q1)
######

source_start = 0.75
source_end   = 1.25

drain_start  = 0.75
drain_end    = 1.25

source_indices =  (q2 > source_start) & (q2 < source_end)
drain_indices  =  (q1 > drain_start)  & (q1 < drain_end )

# Left needs to be near source, right sensor near drain
sensor_1_left_start = 0.75 # um
sensor_1_left_end   = 1.25 # um

sensor_1_right_start = 0.75 # um
sensor_1_right_end   = 1.25 # um

sensor_1_left_indices  = (q1 > sensor_1_left_start ) & (q1 < sensor_1_left_end)
sensor_1_right_indices = (q2 > sensor_1_right_start) & (q2 < sensor_1_right_end)


######

moment_files_1 		  = np.sort(glob.glob(filepath+'/moments_1_*.h5'))
lagrange_multiplier_files_1 = \
        np.sort(glob.glob(filepath+'/lagrange_multipliers_1_*.h5'))

moment_files_2 		  = np.sort(glob.glob(filepath+'/moments_2_*.h5'))
lagrange_multiplier_files_2 = \
        np.sort(glob.glob(filepath+'/lagrange_multipliers_2_*.h5'))

moment_files_3 		  = np.sort(glob.glob(filepath+'/moments_3_*.h5'))
lagrange_multiplier_files_3 = \
        np.sort(glob.glob(filepath+'/lagrange_multipliers_3_*.h5'))

dt = params_1.dt
dump_interval = params_1.dump_steps

N_g = domain_1.N_ghost

time_array = np.loadtxt("dump_time_array.txt")

######

source_array = []
drain_array = []
sensor_1_signal_array = []

for file_number, dump_file in yt.parallel_objects(enumerate(moment_files_1[:])):

    #file_number = -1
    print("file number = ", file_number, "of ", moment_files_1.size)

    h5f  = h5py.File(moment_files_1[file_number], 'r')
    moments_1 = np.swapaxes(h5f['moments'][:], 0, 1)
    h5f.close()

    density_1 = moments_1[:, :, 0]
    j_x_1     = moments_1[:, :, 1]
    j_y_1     = moments_1[:, :, 2]

    h5f  = h5py.File(lagrange_multiplier_files_1[file_number], 'r')
    lagrange_multipliers_1 = h5f['lagrange_multipliers'][:]
    h5f.close()

    mu_1    = lagrange_multipliers_1[:, :, 0]
    mu_ee_1 = lagrange_multipliers_1[:, :, 1]
    T_ee_1  = lagrange_multipliers_1[:, :, 2]
    vel_drift_x_1 = lagrange_multipliers_1[:, :, 3]
    vel_drift_y_1 = lagrange_multipliers_1[:, :, 4]
  
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
    
    h5f  = h5py.File(moment_files_3[file_number], 'r')
    moments_3 = np.swapaxes(h5f['moments'][:], 0, 1)
    h5f.close()

    density_3 = moments_3[:, :, 0]
    j_x_3     = moments_3[:, :, 1]
    j_y_3     = moments_3[:, :, 2]

    h5f  = h5py.File(lagrange_multiplier_files_3[file_number], 'r')
    lagrange_multipliers_3 = h5f['lagrange_multipliers'][:]
    h5f.close()

    mu_3    = lagrange_multipliers_3[:, :, 0]
    mu_ee_3 = lagrange_multipliers_3[:, :, 1]
    T_ee_3  = lagrange_multipliers_3[:, :, 2]
    vel_drift_x_3 = lagrange_multipliers_3[:, :, 3]
    vel_drift_y_3 = lagrange_multipliers_3[:, :, 4]
    
    #print ("density_1 : ", vel_drift_x_1.shape)
    #print ("density_2 : ", vel_drift_x_2.shape)
    shape_x = density_1.shape[0] + density_2.shape[0] + density_3.shape[0]
    shape_y = density_2.shape[1]
    
    combined_density_array     = np.zeros((shape_x, shape_y))

    q2_midpoint   = int(shape_y/2)
    contact_width = int(density_1.shape[1])
    q2_index      = int(q2_midpoint - contact_width/2)

    q1_midpoint  = int(shape_x/2)
    q1_index     = int(q1_midpoint - contact_width/2)

    combined_density_array[:density_1.shape[0], q2_index:q2_index+contact_width] = density_1
    combined_density_array[q1_index:q1_index+contact_width, :] = density_2
    combined_density_array[q1_index+contact_width:, q2_index:q2_index+contact_width] = density_3

    source = np.mean(combined_density_array[0, source_indices])
    drain  = np.mean(combined_density_array[drain_indices, 0])

    source_array.append(source)
    drain_array.append(drain)

    sensor_1_left   = np.mean(combined_density_array[sensor_1_left_indices, -1] )
    sensor_1_right  = np.mean(combined_density_array[-1, sensor_1_right_indices])

    sensor_1_signal = sensor_1_left - sensor_1_right
    sensor_1_signal_array.append(sensor_1_signal)
    
sensor_1_signal_array = np.array(sensor_1_signal_array)
source_array          = np.array(source_array)
drain_array           = np.array(drain_array)

half_time = (int)(time_array.size/2)

#input_normalized = \
#    input_signal_array/np.max(np.abs(input_signal_array[half_time:]))
sensor_1_normalized = \
    sensor_1_signal_array/np.max(np.abs(sensor_1_signal_array[half_time:]))

# Calculate the phase difference between input_signal_array and sensor_normalized
# Code copied from :
# https:/stackoverflow.com/questions/6157791/find-phase-difference-between-two-inharmonic-waves

#corr = correlate(sensor_1_normalized, sensor_2_normalized)
#nsamples = input_normalized.size
#time_corr = time_array[half_time:]
#dt_corr = np.linspace(-time_corr[-1] + time_corr[0],
#                            time_corr[-1] - time_corr[0], 2*nsamples-1)
#time_shift = dt_corr[corr.argmax()]
#print ('Time shift : ', time_shift)

#Force the phase shift to be in [-pi:pi]
#period = 1./AC_freq
#phase_diff = 2*np.pi*(((0.5 + time_shift/period) % 1.0) - 0.5)

pl.rcParams['figure.figsize']  = 12, 7.5

pl.plot(time_array, source_array-drain_array)
pl.plot(time_array, sensor_1_signal_array)
#pl.plot(time_array, sensor_2_normalized)
pl.axhline(0, color='black', linestyle='--')

pl.legend(['Source $I(t)$', 'Measured $V_{1}(t)$'], loc=1)
#pl.text(135, 1.14, '$\phi : %.3f \; rad$' %phase_diff)
pl.xlabel(r'Time (ps)')
#pl.xlim([0, 100])
#pl.ylim([-1.1, 1.1])

pl.suptitle('$\\tau_\mathrm{mc} = \infty$ ps, $\\tau_\mathrm{mr} = 1000.0$ ps')
pl.savefig('images/iv' + '.png')
pl.clf()
    

