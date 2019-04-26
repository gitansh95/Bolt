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
moment_files_4 = np.sort(glob.glob(filepath+'/ping_pong/dumps_4_r/moment*.h5'))
time_array_mirror = np.loadtxt(filepath+"/ping_pong/dump_time_array.txt")

filepath = \
'/home/mchandra/gitansh/gitansh_bolt/example_problems/electronic_boltzmann'
moment_files_8 = np.sort(glob.glob(filepath+'/ping_pong/dumps_8_r/moment*.h5'))
time_array_mirror = np.loadtxt(filepath+"/ping_pong/dump_time_array.txt")

filepath = \
'/home/mchandra/gitansh/gitansh_bolt/example_problems/electronic_boltzmann'
moment_files_16 = np.sort(glob.glob(filepath+'/ping_pong/dumps_16_r/moment*.h5'))
time_array_mirror = np.loadtxt(filepath+"/ping_pong/dump_time_array.txt")

filepath = \
'/home/mchandra/gitansh/gitansh_bolt/example_problems/electronic_boltzmann'
moment_files_64 = np.sort(glob.glob(filepath+'/ping_pong/dumps_64_r/moment*.h5'))
time_array_mirror = np.loadtxt(filepath+"/ping_pong/dump_time_array.txt")

filepath = \
'/home/mchandra/gitansh/gitansh_bolt/example_problems/electronic_boltzmann'
moment_files_128 = np.sort(glob.glob(filepath+'/ping_pong/dumps_128_r/moment*.h5'))
time_array_mirror = np.loadtxt(filepath+"/ping_pong/dump_time_array.txt")

filepath = \
'/home/mchandra/gitansh/gitansh_bolt/example_problems/electronic_boltzmann'
moment_files_1024 = np.sort(glob.glob(filepath+'/ping_pong/dumps_1024_r/moment*.h5'))
time_array_mirror = np.loadtxt(filepath+"/ping_pong/dump_time_array.txt")


integrated_density_4 = []
for file_number, dump_file in enumerate(moment_files_4):


    h5f  = h5py.File(dump_file, 'r')
    moments = np.swapaxes(h5f['moments'][:], 0, 1)
    h5f.close()

    density = moments[:, :, 0]
    integrated_density_4.append(np.sum(density))

integrated_density_8 = []
for file_number, dump_file in enumerate(moment_files_8):


    h5f  = h5py.File(dump_file, 'r')
    moments = np.swapaxes(h5f['moments'][:], 0, 1)
    h5f.close()

    density = moments[:, :, 0]
    integrated_density_8.append(np.sum(density))

integrated_density_16 = []
for file_number, dump_file in enumerate(moment_files_16):


    h5f  = h5py.File(dump_file, 'r')
    moments = np.swapaxes(h5f['moments'][:], 0, 1)
    h5f.close()

    density = moments[:, :, 0]
    integrated_density_16.append(np.sum(density))

integrated_density_64 = []
for file_number, dump_file in enumerate(moment_files_64):


    h5f  = h5py.File(dump_file, 'r')
    moments = np.swapaxes(h5f['moments'][:], 0, 1)
    h5f.close()

    density = moments[:, :, 0]
    integrated_density_64.append(np.sum(density))

integrated_density_128 = []
for file_number, dump_file in enumerate(moment_files_128):


    h5f  = h5py.File(dump_file, 'r')
    moments = np.swapaxes(h5f['moments'][:], 0, 1)
    h5f.close()

    density = moments[:, :, 0]
    integrated_density_128.append(np.sum(density))

integrated_density_1024 = []
for file_number, dump_file in enumerate(moment_files_1024):


    h5f  = h5py.File(dump_file, 'r')
    moments = np.swapaxes(h5f['moments'][:], 0, 1)
    h5f.close()

    density = moments[:, :, 0]
    integrated_density_1024.append(np.sum(density))

integrated_density_4 = np.array(integrated_density_4)
integrated_density_8 = np.array(integrated_density_8)
integrated_density_16 = np.array(integrated_density_16)
integrated_density_64 = np.array(integrated_density_64)
integrated_density_128 = np.array(integrated_density_128)
integrated_density_1024 = np.array(integrated_density_1024)

pl.plot(time_array_mirror, integrated_density_4, '-o', label='N=4')
pl.plot(time_array_mirror, integrated_density_8, '-o', label='N=8')
pl.plot(time_array_mirror, integrated_density_16, '-o', label='N=16')
pl.plot(time_array_mirror, integrated_density_64, '-o', label='N=64')
pl.plot(time_array_mirror, integrated_density_128, '-o', label='N=128')
pl.plot(time_array_mirror, integrated_density_1024, '-o', label='N=1024')
#pl.plot(time_array_periodic, integrated_density_periodic, '-o', label='Periodic')
pl.xlabel(r'$\mathrm{Time\ (ps)}$')
pl.ylabel(r'$\mathrm{\Sigma n}$')
#pl.ylim([0.000076, 0.000078])

pl.legend(loc = 'best')

pl.savefig('images/density.png')
pl.clf()
    
