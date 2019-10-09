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

# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 8, 8
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

import PetscBinaryIO
io = PetscBinaryIO.PetscBinaryIO()

q1_start = 0.0
q1_end = 1.0

q2_start = 0.0
q2_end = 5.0

N_q1 = 72
N_q2 = int(round(N_q1*q2_end))

q1 = q1_start + (0.5 + np.arange(N_q1)) * (q1_end - q1_start)/N_q1
q2 = q2_start + (0.5 + np.arange(N_q2)) * (q2_end - q2_start)/N_q2

q2_meshgrid, q1_meshgrid = np.meshgrid(q2, q1)

N_p1 = 1
N_p2 = 8192

p1_start = 0.5
p1_end = 1.5

p2_start =  -3.14159265359
p2_end   =  3.14159265359

p1 = p1_start + (0.5 + np.arange(N_p1)) * (p1_end - p1_start)/N_p1
p2 = p2_start + (0.5 + np.arange(N_p2)) * (p2_end - p2_start)/N_p2

#p2_meshgrid, p1_meshgrid = np.meshgrid(p2, p1)

filepath = 'L_1.0_5.0_tau_ee_inf_tau_eph_10.0'

moment_files 		        = np.sort(glob.glob(filepath+'/moment*.h5'))
lagrange_multiplier_files   = \
        np.sort(glob.glob(filepath+'/lagrange_multipliers*.h5'))
distribution_function_files = np.sort(glob.glob(filepath+'/f_*.h5'))

file_number = moment_files.size-1

dt = 0.025/2
dump_interval = 5

#time_array = np.loadtxt("dump_time_array.txt")
moments_file = moment_files[file_number]
moments = io.readBinaryFile(moments_file)
print (np.ndim(moments))
moments = moments[0].reshape(N_q2, N_q1, 3)


dist_func_bg_file = distribution_function_files[0]
dist_func_bg = io.readBinaryFile(dist_func_bg_file)
print (dist_func_bg_file)
print (np.ndim(dist_func_bg))
dist_func_bg = dist_func_bg[0].reshape(N_q2, N_q1, 1, N_p2, N_p1)

dist_func_file = distribution_function_files[-1]
dist_func = io.readBinaryFile(dist_func_file)
dist_func = dist_func_bg[0].reshape(N_q2, N_q1, 1, N_p2, N_p1)



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
        
        np.savetxt('f_at_a_point_L_1.0_5.0_tau_ee_inf_tau_eph_10.0_%d_%d.png'%(index_1, index_2))
        #im = pl.plot(p2, f_at_desired_q.transpose())i
        #pl.title(r'Time = ' + "%.2f"%(time_array[file_number]) + " ps")
        #pl.ylabel('$f$')
        #pl.xlabel('$p_{\\theta}$')
        #pl.tight_layout()
        #pl.savefig('images/dist_func_at_a_point_%d_%d.png'%(index_1, index_2))
        #pl.clf()


f_spatial_avg = np.reshape(np.mean(dist_func-dist_func_background, axis=(0,1)), [N_p1, N_p2])

pl.plot(p2, f_spatial_avg.transpose())
#pl.title(r'Time = ' + "%.2f"%(time_array[file_number]) + " ps")
pl.ylabel('$f$')
pl.xlabel('$p_{\\theta}$')
#pl.tight_layout()
pl.savefig('images/dist_func_spatial_avg.png')
pl.clf()

# Real space plot
moments_file = moment_files[file_number]
moments = io.readBinaryFile(moments_file)
moments = moments[0].reshape(N_q2, N_q1, 3)

density = moments[:, :, 0]
j_x     = moments[:, :, 1]
j_y     = moments[:, :, 2]

lagrange_multipliers_file = lagrange_multiplier_files[file_number]
lagrange_multipliers = io.readBinaryFile(lagrange_multipliers_file)
lagrange_multipliers = lagrange_multipliers[0].reshape(N_q2, N_q1, 7)

mu    = lagrange_multipliers[:, :, 0]
mu_ee = lagrange_multipliers[:, :, 1]
T_ee  = lagrange_multipliers[:, :, 2]
vel_drift_x = lagrange_multipliers[:, :, 3]
vel_drift_y = lagrange_multipliers[:, :, 4]

pl.subplot(1,2,1)
pl.contourf(q1_meshgrid, q2_meshgrid, density, 100, cmap='bwr')
#pl.title(r'Time = ' + "%.2f"%(time_array[file_number]) + " ps")
pl.streamplot(q1, q2, 
              vel_drift_x, vel_drift_y,
              density=2, color='blue',
              linewidth=0.7, arrowsize=1
             )

pl.xlim([q1_start, q1_end])
pl.ylim([q2_start, q2_end])

pl.gca().set_aspect('equal')
pl.xlabel(r'$x\;(\mu \mathrm{m})$')
pl.ylabel(r'$y\;(\mu \mathrm{m})$')

pl.subplot(1,2,2)
pl.contourf(q1_meshgrid, q2_meshgrid, density, 100, cmap='bwr')
#pl.title(r'Time = ' + "%.2f"%(time_array[file_number]) + " ps")
for index_1 in range(N):
    for index_2 in range(N):
        q1_position = int(N_q1*((index_1/N)+(1/(2*N))))
        q2_position = int(N_q2*((index_2/N)+(1/(2*N))))
        txt = "(%d, %d)"%(index_1, index_2)
        pl.plot(q1[q1_position], q2[q2_position], color = 'k', marker = 'o',
                markersize = 3)
        pl.text(q1[q1_position], q2[q2_position], txt, fontsize=8)

pl.xlim([q1_start, q1_end])
#pl.ylim([q2_start, q2_end])

pl.gca().set_aspect('equal')
pl.xlabel(r'$x\;(\mu \mathrm{m})$')
pl.ylabel(r'$y\;(\mu \mathrm{m})$')

pl.suptitle('$\\tau_\mathrm{mc} = \infty$ ps, $\\tau_\mathrm{mr} = 10.0$ ps')
pl.savefig('f_data/dump_last_step.png')
pl.clf()
