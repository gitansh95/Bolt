import arrayfire as af
import numpy as np
from scipy.signal import correlate
import glob
import h5py
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib import transforms, colors
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
sensor_1_left_start = 3.5 # um
sensor_1_left_end   = 4.5 # um

sensor_1_right_start = 3.5 # um
sensor_1_right_end   = 4.5 # um

sensor_1_left_indices  = (q2 > sensor_1_left_start ) & (q2 < sensor_1_left_end)
sensor_1_right_indices = (q2 > sensor_1_right_start) & (q2 < sensor_1_right_end)

sensor_2_left_start = 0.0 # um
sensor_2_left_end   = 0.25 # um

sensor_2_right_start = 0.0 # um
sensor_2_right_end   = 0.25 # um

sensor_2_left_indices  = (q2 > sensor_2_left_start ) & (q2 < sensor_2_left_end)
sensor_2_right_indices = (q2 > sensor_2_right_start) & (q2 < sensor_2_right_end)

filepath = \
'/home/mchandra/gitansh/gitansh_bolt/example_problems/electronic_boltzmann/graphene/L_2.0_20.0_tau_ee_inf_tau_eph_100.00_DC_half/dumps'
moment_files 		      = np.sort(glob.glob(filepath+'/moment*.h5'))
lagrange_multiplier_files = \
        np.sort(glob.glob(filepath+'/lagrange_multipliers*.h5'))

dt = params.dt
dump_interval = params.dump_steps

time_array = np.loadtxt("dump_time_array.txt")

for file_number, dump_file in yt.parallel_objects(enumerate(moment_files)):

    h5f  = h5py.File(dump_file, 'r')
    moments = np.swapaxes(h5f['moments'][:], 0, 1)
    h5f.close()

    print("file number = ", file_number, "of ", moment_files.size, "shape = ", moments.shape)

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
    
    
    delta_n = density - np.mean(density)
    colorlevels = np.linspace(-delta_n.max()/5, delta_n.max()/5, 200)

    j_mag = (j_x**2. + j_y**2.)**0.5

    indices = q2_meshgrid > 1.

    pl.figure(figsize=(25, 7.5))

    base = pl.gca().transData
    rot = transforms.Affine2D().rotate_deg(-90)

    pl.contourf(q1_meshgrid, q2_meshgrid, delta_n, 200,
            norm=colors.SymLogNorm(linthresh=delta_n.max()/20), cmap='bwr',
            transform = rot + base)
    #pl.contourf(q1_meshgrid, q2_meshgrid, delta_n, 200, cmap='bwr', transform =
    #rot + base)

    pl.streamplot(q1, q2, vel_drift_x, vel_drift_y,
                          density=5, color='blue',
                                        linewidth=0.7, arrowsize=1, transform =
                                        rot + base
                                                     )

    pl.xlim([0, 5])
    pl.ylim([-1, 0])
    pl.yticks([])
    #pl.xlim([q1_start, q1_end])
    #pl.ylim([q2_start, q2_end])
    pl.gca().set_aspect('equal')
    pl.title(r'Time = ' + "%.2f"%(time_array[file_number]) + " ps")

    pl.suptitle('$\\tau_\mathrm{mc} = \infty$ ps, $\\tau_\mathrm{mr} = 100.0$ ps')
    pl.savefig('images/dump_' + '%06d'%file_number + '.png')
    pl.clf()
    
