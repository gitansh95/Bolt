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
'/home/mchandra/gitansh/gitansh_bolt/example_problems/electronic_boltzmann'
moment_files = \
    np.sort(glob.glob(filepath+'/L_1.0_2.5_classical_ballistic_electrons/dumps/moment*.h5'))
time_array_mirror = \
    np.loadtxt(filepath+"/L_1.0_2.5_classical_ballistic_electrons/dump_time_array.txt")

moment_files_h = \
    np.sort(glob.glob(filepath+'/L_1.0_2.5_classical_ballistic_holes/dumps/moment*.h5'))
time_array_mirror_h = \
    np.loadtxt(filepath+"/L_1.0_2.5_classical_ballistic_holes/dump_time_array.txt")


integrated_density = []
integrated_density_h = []
integrated_density_combined = []
for file_number, dump_file in enumerate(moment_files):


    h5f  = h5py.File(dump_file, 'r')
    moments = np.swapaxes(h5f['moments'][:], 0, 1)
    h5f.close()
    h5f  = h5py.File(moment_files_h[file_number], 'r')
    moments_h = np.swapaxes(h5f['moments'][:], 0, 1)
    h5f.close()

    density = moments[:, :, 0]
    density_h = moments_h[:, :, 0]
    density_combined = density + density_h

    integrated_density.append(np.sum(density))
    integrated_density_h.append(np.sum(density_h))
    integrated_density_combined.append(np.sum(density_combined))

integrated_density = np.array(integrated_density)
integrated_density_h = np.array(integrated_density_h)
integrated_density_combined = np.array(integrated_density_combined)

pl.plot(time_array_mirror, integrated_density, label='e')
pl.plot(time_array_mirror_h, integrated_density_h, label='h')
pl.plot(time_array_mirror, integrated_density_combined, label='e+h')

np.savetxt("density_e.txt", integrated_density)
np.savetxt("density_h.txt", integrated_density_h)


#pl.plot(time_array_periodic, integrated_density_periodic, '-o', label='Periodic')
pl.xlabel(r'$\mathrm{Time\ (ps)}$')
pl.ylabel(r'$\mathrm{\Sigma n}$')
#pl.ylim([0.000076, 0.000078])

pl.legend(loc = 'best')

pl.savefig('images/density.png')
pl.clf()
    
