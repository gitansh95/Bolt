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

dq1 = (domain.q1_end - domain.q1_start)/N_q1
dq2 = (domain.q2_end - domain.q2_start)/N_q2

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
'/home/mchandra/gitansh/zero_T_with_mirror/example_problems/electronic_boltzmann/graphene/L_1.0_1.0_tau_ee_inf_tau_eph_1000.0_DC_L_bent/dumps'
#'/root/bolt/Bolt/example_problems/electronic_boltzmann/graphene/L_1.0_1.0_tau_ee_inf_tau_eph_1000.0_DC_L_bent/dumps'
moment_files 		  = np.sort(glob.glob(filepath+'/moment*.h5'))
lagrange_multiplier_files = \
        np.sort(glob.glob(filepath+'/lagrange_multipliers*.h5'))

dt = params.dt
dump_interval = params.dump_steps

N_g = domain.N_ghost

time_array = np.loadtxt("dump_time_array.txt")

for file_number, dump_file in yt.parallel_objects(enumerate(moment_files[:])):

    #file_number = -1
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
  
    print("density shape : ", density.shape)
    print("vel_y.shape = ", vel_drift_y.shape)
   
    pl.subplot(1,2,1)
    pl.contourf(q1_meshgrid, q2_meshgrid, density, 100, cmap='bwr')
    pl.title(r'Time = ' + "%.2f"%(time_array[file_number]) + " ps")

    pl.streamplot(q1, q2, 
                  vel_drift_x, vel_drift_y,
                  density=2, color='black',
                  linewidth=0.7, arrowsize=1
                 )
    
    pl.xlim([domain.q1_start, domain.q1_end])
    pl.ylim([domain.q2_start, domain.q2_end])

    pl.gca().set_xticks(q1 - 0.5*dq1)
    pl.gca().set_yticks(q2 - 0.5*dq2)

    pl.grid('on')
    pl.plot(q1_meshgrid, q2_meshgrid, marker='.', markersize = 2, color='k', linestyle='none')

    if (params.horizontal_internal_bcs_enabled):
        mirror_indices = q1[((q1 >= params.horizontal_mirror_0_start) & \
                            (q1 <= params.horizontal_mirror_0_end))]


        pl.axhline(0.25, color = 'k', ls = '--')
        pl.axhspan(ymin = q2[params.horizontal_mirror_0_index - 2*N_g] - dq2/2,
                   ymax = q2[params.horizontal_mirror_0_index] - dq2/2,
                   xmin = (mirror_indices[0]-dq1/2)/domain.q1_end,
                   xmax = (mirror_indices[-1]+dq1/2)/domain.q1_end,
                   color='k', alpha = 0.5)
        
        #print ("Horizontal : ", mirror_indices)
        #print (q2[params.horizontal_mirror_0_index - 2*N_g])
        #print (q2[params.horizontal_mirror_0_index])
    
    if (params.vertical_internal_bcs_enabled):
        mirror_indices = q2[((q2 >= params.vertical_mirror_0_start) & \
                            (q2 <= params.vertical_mirror_0_end))]

        pl.axvline(0.75, color = 'k', ls = '--')
        pl.axvspan(xmin = q1[params.vertical_mirror_0_index - 2*N_g] - dq1/2,
                   xmax = q1[params.vertical_mirror_0_index] - dq1/2,
                   ymin = (mirror_indices[0]-dq2/2)/domain.q2_end,
                   ymax = (mirror_indices[-1]+dq2/2)/domain.q2_end,
                   color='k', alpha = 0.5)
        
        #print ("Vertical : ", mirror_indices)
        #print (q1[params.vertical_mirror_0_index - 2*N_g])
        #print (q1[params.vertical_mirror_0_index])

    
    pl.gca().set_aspect('equal')
    pl.xlabel(r'$x\;(\mu \mathrm{m})$')
    pl.ylabel(r'$y\;(\mu \mathrm{m})$')

    pl.subplot(1,2,2)
    pl.contourf(q1_meshgrid, q2_meshgrid, density, 100, cmap='bwr')
    pl.title(r'Time = ' + "%.2f"%(time_array[file_number]) + " ps")

    pl.xlim([domain.q1_start, domain.q1_end])
    pl.ylim([domain.q2_start, domain.q2_end])
    
    pl.gca().set_aspect('equal')
    pl.xlabel(r'$x\;(\mu \mathrm{m})$')
    #pl.ylabel(r'$y\;(\mu \mathrm{m})$')

    pl.suptitle('$\\tau_\mathrm{mc} = \infty$ ps, $\\tau_\mathrm{mr} = 5.0$ ps')
    pl.savefig('images/dump_' + '%06d'%file_number + '.png')
    pl.clf()
    
