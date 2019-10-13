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
from mpl_toolkits.mplot3d import Axes3D

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
    
N_p1 = domain.N_p1
N_p2 = domain.N_p2

p1 = domain.p1_start + (0.5 + np.arange(N_p1)) * (domain.p1_end - domain.p1_start)/N_p1
p2 = domain.p2_start + (0.5 + np.arange(N_p2)) * (domain.p2_end - domain.p2_start)/N_p2

#p2_meshgrid, p1_meshgrid = np.meshgrid(p2, p1)
    
N = 7
index_1 = 1
index_2 = 0

q1_position = int(domain.N_q1*((index_1/N)+(1/(2*N))))
q2_position = int(domain.N_q2*((index_2/N)+(1/(2*N))))


filepath = \
    '/home/mchandra/gitansh/gitansh_bolt/example_problems/electronic_boltzmann/L_5.0_10.0_classical_ballistic_angular_spread/dumps'
distribution_function_files = np.sort(glob.glob(filepath+'/f_*.h5'))
    
time_array = np.loadtxt(filepath+"/../dump_time_array.txt")

h5f  = h5py.File(distribution_function_files[0], 'r')
dist_func_background = h5f['distribution_function'][:]
h5f.close()



print ('Files : ', distribution_function_files.size)


for file_number in range(distribution_function_files.size):
    print ("File number : ", file_number)
    
    h5f  = h5py.File(distribution_function_files[file_number], 'r')
    dist_func = h5f['distribution_function'][:]
    h5f.close()

    norm_factor = np.max(np.abs((dist_func - dist_func_background))[q2_position, q1_position, :])
    f_at_desired_q = \
        np.reshape((dist_func-dist_func_background)[q2_position, q1_position,:], [N_p1, N_p2])/norm_factor
    
    np.savetxt('data/f_vs_theta_%d_%d_%06d.txt'%(index_1, index_2,
        file_number), f_at_desired_q)
    f = np.loadtxt('data/f_vs_theta_%d_%d_%06d.txt'%(index_1, index_2,
        file_number))
    
    radius = f.copy()
    theta  = p2.copy()

    x = (radius + 5.)*np.cos(theta)
    y = (radius + 5.)*np.sin(theta)
    
    x_bg = 5*np.cos(theta)
    y_bg = 5*np.sin(theta)

    #pl.subplot(121)
    pl.plot(x, y, color='r', linestyle = '-')
    pl.plot(x_bg, y_bg, color='k', alpha=0.2)
    pl.title(r'Time = ' + "%.2f"%(time_array[file_number]) + ' ps')
    #ax.set_ylabel('$f$')
    #ax.set_xlabel('$p_{\\theta}$')
    pl.gca().set_aspect('equal')
    pl.xlim([-6.5, 6.5])
    pl.ylim([-6.5, 6.5])

    pl.savefig('images/fermi_surface_at_a_point_%d_%d_%06d.png'%(index_1, index_2, file_number))
    pl.clf()


