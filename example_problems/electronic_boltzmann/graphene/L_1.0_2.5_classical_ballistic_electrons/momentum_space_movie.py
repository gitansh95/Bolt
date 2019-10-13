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

N_p1 = domain.N_p1
N_p2 = domain.N_p2

p1 = domain.p1_start + (0.5 + np.arange(N_p1)) * (domain.p1_end - domain.p1_start)/N_p1
p2 = domain.p2_start + (0.5 + np.arange(N_p2)) * (domain.p2_end - domain.p2_start)/N_p2

#p2_meshgrid, p1_meshgrid = np.meshgrid(p2, p1)

filepath = \
'/home/mchandra/gitansh/gitansh_bolt/example_problems/electronic_boltzmann/L_1.0_2.5_classical_ballistic_electrons/dumps'
moment_files 		        = np.sort(glob.glob(filepath+'/moment*.h5'))
lagrange_multiplier_files   = \
        np.sort(glob.glob(filepath+'/lagrange_multipliers*.h5'))
distribution_function_files = np.sort(glob.glob(filepath+'/f_*.h5'))

filepath_2 = \
'/home/mchandra/gitansh/gitansh_bolt/example_problems/electronic_boltzmann/L_1.0_2.5_classical_ballistic_holes/dumps'
#moment_files 		        = np.sort(glob.glob(filepath+'/moment*.h5'))
#lagrange_multiplier_files   = \
#        np.sort(glob.glob(filepath+'/lagrange_multipliers*.h5'))
distribution_function_files_2 = np.sort(glob.glob(filepath_2+'/f_*.h5'))

dt = params.dt
dump_interval = params.dump_steps

time_array = np.loadtxt("dump_time_array.txt")

h5f  = h5py.File(distribution_function_files[0], 'r')
dist_func_background = h5f['distribution_function'][:]
h5f.close()

h5f  = h5py.File(distribution_function_files[-1], 'r')
dist_func = h5f['distribution_function'][:]
h5f.close()

h5f  = h5py.File(distribution_function_files_2[-1], 'r')
dist_func_2 = h5f['distribution_function'][:]
h5f.close()

file_number = moment_files.size-1

N = 7
for index_1 in range(N):
    for index_2 in range(N):

        q1_position = int(domain.N_q1*((index_1/N)+(1/(2*N))))
        q2_position = int(domain.N_q2*((index_2/N)+(1/(2*N))))
        
        a = np.max((dist_func - dist_func_background)[q2_position, q1_position, :])
        b = np.abs(np.min((dist_func - dist_func_background)[q2_position, q1_position, :]))
        norm_factor = np.maximum(a, b)
        f_at_desired_q = \
        np.reshape((dist_func-dist_func_background)[q2_position, q1_position,:], [N_p1, N_p2])/norm_factor
    
        np.savetxt('data/f_vs_theta_%d_%d_%06d.txt'%(index_1, index_2,
            file_number), f_at_desired_q)
        f = np.loadtxt('data/f_vs_theta_%d_%d_%06d.txt'%(index_1, index_2,
            file_number))
        
        a = np.max((dist_func_2 - dist_func_background)[q2_position, q1_position, :])
        b = np.abs(np.min((dist_func_2 - dist_func_background)[q2_position, q1_position, :]))
        norm_factor = np.maximum(a, b)
        f_at_desired_q = \
        np.reshape((dist_func_2-dist_func_background)[q2_position, q1_position,:], [N_p1, N_p2])/norm_factor
    
        np.savetxt('data/f_vs_theta_2_%d_%d_%06d.txt'%(index_1, index_2,
            file_number), f_at_desired_q)
        f_2 = np.loadtxt('data/f_vs_theta_2_%d_%d_%06d.txt'%(index_1, index_2,
            file_number))
        
        radius = f.copy()
        radius_2 = f_2.copy()
        radius_3 = radius + radius_2
        theta  = p2.copy()

        x = (radius + 5.)*np.cos(theta)
        y = (radius + 5.)*np.sin(theta)

        x_2 = (radius_2 + 5.)*np.cos(theta)
        y_2 = (radius_2 + 5.)*np.sin(theta)
        
        x_3 = (radius_3 + 5.)*np.cos(theta)
        y_3 = (radius_3 + 5.)*np.sin(theta)
        
        x_bg = 5*np.cos(theta)
        y_bg = 5*np.sin(theta)

        #pl.subplot(121)
        #pl.plot(x, y, color='C0', linestyle = '-', alpha = 0.3)
        #pl.plot(x_2, y_2, color='C1', linestyle = '-', alpha = 0.3)
        pl.plot(x_3, y_3, color='r', linestyle = '-')
        pl.plot(x_bg, y_bg, color='k', alpha=0.4)
        #pl.title(r'Time = ' + "%.2f"%(time_array[file_number]) + ' ps')
        #ax.set_ylabel('$f$')
        #ax.set_xlabel('$p_{\\theta}$')

        pl.gca().set_xticklabels([])
        pl.gca().set_yticklabels([])

        pl.gca().set_aspect('equal')
        pl.xlim([-6.3, 6.3])
        pl.ylim([-6.3, 6.3])
        
        pl.savefig('images/dist_func_at_a_point_%d_%d.png'%(index_1, index_2), transparent = True)
        pl.clf()


f_spatial_avg = np.reshape(np.mean(dist_func-dist_func_background, axis=(0,1)), [N_p1, N_p2])

pl.plot(p2, f_spatial_avg.transpose())
pl.title(r'Time = ' + "%.2f"%(time_array[file_number]) + " ps")
pl.ylabel('$f$')
pl.xlabel('$p_{\\theta}$')
#pl.tight_layout()
pl.savefig('images/dist_func_spatial_avg.png')
pl.clf()

# Real space plot
h5f  = h5py.File(moment_files[file_number], 'r')
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

pl.subplot(1,2,1)
pl.contourf(q1_meshgrid, q2_meshgrid, density, 100, cmap='bwr')
pl.title(r'Time = ' + "%.2f"%(time_array[file_number]) + " ps")
pl.streamplot(q1, q2, 
              vel_drift_x, vel_drift_y,
              density=2, color='blue',
              linewidth=0.7, arrowsize=1
             )

pl.xlim([domain.q1_start, domain.q1_end])
pl.ylim([domain.q2_start, domain.q2_end])

pl.gca().set_aspect('equal')
pl.xlabel(r'$x\;(\mu \mathrm{m})$')
pl.ylabel(r'$y\;(\mu \mathrm{m})$')

pl.subplot(1,2,2)
pl.contourf(q1_meshgrid, q2_meshgrid, density, 100, cmap='bwr')
pl.title(r'Time = ' + "%.2f"%(time_array[file_number]) + " ps")
for index_1 in range(N):
    for index_2 in range(N):
        q1_position = int(domain.N_q1*((index_1/N)+(1/(2*N))))
        q2_position = int(domain.N_q2*((index_2/N)+(1/(2*N))))
        txt = "(%d, %d)"%(index_1, index_2)
        pl.plot(q1[q1_position], q2[q2_position], color = 'k', marker = 'o',
                markersize = 3)
        pl.text(q1[q1_position], q2[q2_position], txt, fontsize=8)

pl.xlim([domain.q1_start, domain.q1_end])
#pl.ylim([domain.q2_start, domain.q2_end])

pl.gca().set_aspect('equal')
pl.xlabel(r'$x\;(\mu \mathrm{m})$')
pl.ylabel(r'$y\;(\mu \mathrm{m})$')

#pl.suptitle('$\\tau_\mathrm{mc} = \infty$ ps, $\\tau_\mathrm{mr} = 9.0$ ps')
pl.savefig('images/dump_last_step.png', transparent = True)
pl.clf()
