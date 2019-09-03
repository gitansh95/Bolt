import arrayfire as af
import numpy as np
from scipy.signal import correlate
import glob
import h5py
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.colors as colors

matplotlib.use('agg')
import pylab as pl
import yt
yt.enable_parallelism()

import petsc4py, sys; petsc4py.init(sys.argv)
from petsc4py import PETSc

import domain_1
import boundary_conditions_1
import params_1
import initialize_1

import domain_2
import boundary_conditions_2
import params_2
import initialize_2

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

class MidpointNormalize (colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
    # I'm ignoring masked values and all kinds of edge cases to make
    # a simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


filepath = \
'/home/mchandra/gitansh/zero_T_with_mirror/example_problems/electronic_boltzmann/graphene/L_1.0_1.0_dual_domain/dumps'
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

N_q1 = N_q1_1
N_q2 = N_q2_1 + N_q2_2

q1_start = 0.
q1_end   = domain_1.q1_end
q2_start = 0.
q2_end   = domain_1.q2_end + domain_2.q2_end

dq1 = (q1_end - q1_start)/N_q1
dq2 = (q2_end - q2_start)/N_q2

q1 = 0. + (0.5 + np.arange(N_q1)) * dq1
q2 = 0. + (0.5 + np.arange(N_q2)) * dq2

q2_meshgrid, q1_meshgrid = np.meshgrid(q2, q1)


moment_files_1 		  = np.sort(glob.glob(filepath+'/moments_1_*.h5'))
lagrange_multiplier_files_1 = \
        np.sort(glob.glob(filepath+'/lagrange_multipliers_1_*.h5'))

moment_files_2 		  = np.sort(glob.glob(filepath+'/moments_2_*.h5'))
lagrange_multiplier_files_2 = \
        np.sort(glob.glob(filepath+'/lagrange_multipliers_2_*.h5'))

dt = params_1.dt
dump_interval = params_1.dump_steps

N_g = domain_1.N_ghost

mean_density = 0
time_array = np.loadtxt("dump_time_array.txt")

for file_number, dump_file in yt.parallel_objects(enumerate(moment_files_1[:1])):
#for file_number in range(1):

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
    
    
    #print ("density_1 : ", vel_drift_x_1.shape)
    #print ("density_2 : ", vel_drift_x_2.shape)
    shape_x = density_1.shape[0]
    shape_y = density_1.shape[1] + density_2.shape[1]
    combined_density_array = np.zeros((shape_x, shape_y))
    combined_vel_drift_x_array = np.zeros((shape_y, shape_x))
    combined_vel_drift_y_array = np.zeros((shape_y, shape_x))

    print ("combined array : ", combined_density_array.shape)
    print ("vel x1 : ", density_1.shape)
    print ("vel x2 : ", density_2.shape)

    combined_density_array[:density_1.shape[0], :density_1.shape[1]] = density_1

    combined_density_array[-density_2.shape[0]:,
            density_1.shape[1]:density_1.shape[1] + density_2.shape[1]] = density_2

    combined_vel_drift_x_array[:vel_drift_x_1.shape[0], :vel_drift_x_1.shape[1]] = \
            vel_drift_x_1
    combined_vel_drift_x_array[vel_drift_x_1.shape[0]:vel_drift_x_1.shape[0] + vel_drift_x_2.shape[0]:, 
            -vel_drift_x_2.shape[1]:] = \
            vel_drift_x_2

    combined_vel_drift_y_array[:vel_drift_x_1.shape[0], :vel_drift_x_1.shape[1]] = \
            vel_drift_y_1
    combined_vel_drift_y_array[vel_drift_x_1.shape[0]:vel_drift_x_1.shape[0] + vel_drift_x_2.shape[0]:, 
            -vel_drift_x_2.shape[1]:] = \
            vel_drift_y_2

    if (file_number == 0):
        mean_density_1 = np.mean(combined_density_array[:density_1.shape[0], :density_1.shape[1]])
        mean_density_2 = np.mean((combined_density_array[-density_2.shape[0]:,
            density_1.shape[1]:density_1.shape[1]+density_2.shape[1]]))
        mean_density = (mean_density_1 + mean_density_2)/2

for file_number, dump_file in yt.parallel_objects(enumerate(moment_files_1[::-1])):
#for file_number in range(1):

    file_number = -1
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
    
    
    #print ("density_1 : ", vel_drift_x_1.shape)
    #print ("density_2 : ", vel_drift_x_2.shape)
    shape_x = density_1.shape[0]
    shape_y = density_1.shape[1] + density_2.shape[1]
    combined_density_array = np.zeros((shape_x, shape_y))
    combined_vel_drift_x_array = np.zeros((shape_y, shape_x))
    combined_vel_drift_y_array = np.zeros((shape_y, shape_x))

    print ("combined array : ", combined_density_array.shape)
    print ("vel x1 : ", density_1.shape)
    print ("vel x2 : ", density_2.shape)

    combined_density_array[:density_1.shape[0], :density_1.shape[1]] = density_1

    combined_density_array[-density_2.shape[0]:,
            density_1.shape[1]:density_1.shape[1] + density_2.shape[1]] = density_2

    combined_vel_drift_x_array[:vel_drift_x_1.shape[0], :vel_drift_x_1.shape[1]] = \
            vel_drift_x_1
    combined_vel_drift_x_array[vel_drift_x_1.shape[0]:vel_drift_x_1.shape[0] + vel_drift_x_2.shape[0]:, 
            -vel_drift_x_2.shape[1]:] = \
            vel_drift_x_2

    combined_vel_drift_y_array[:vel_drift_x_1.shape[0], :vel_drift_x_1.shape[1]] = \
            vel_drift_y_1
    combined_vel_drift_y_array[vel_drift_x_1.shape[0]:vel_drift_x_1.shape[0] + vel_drift_x_2.shape[0]:, 
            -vel_drift_x_2.shape[1]:] = \
            vel_drift_y_2

    if (file_number == 0):
        mean_density_1 = np.mean(combined_density_array[:density_1.shape[0], :density_1.shape[1]])
        mean_density_2 = np.mean((combined_density_array[-density_2.shape[0]:,
            density_1.shape[1]:density_1.shape[1]+density_2.shape[1]]))
        mean_density = (mean_density_1 + mean_density_2)/2
    print ("mean density : ", mean_density)
    combined_density_array[:density_1.shape[0], :density_1.shape[1]] = \
    combined_density_array[:density_1.shape[0], :density_1.shape[1]] - \
    mean_density

    combined_density_array[-density_2.shape[0]:, density_1.shape[1]:density_1.shape[1]+density_2.shape[1]] = \
    combined_density_array[-density_2.shape[0]:, density_1.shape[1]:density_1.shape[1]+density_2.shape[1]] - \
    mean_density
    
    norm = 1.#np.max(np.abs(combined_density_array))
    density_min = np.min(combined_density_array)
    density_max = np.max(combined_density_array)

    im = pl.contourf(q1_meshgrid, q2_meshgrid,
            combined_density_array, 100, 
            norm = MidpointNormalize(midpoint = 0, vmin=density_min, vmax=density_max), cmap='bwr')
    #im.set_clim(vmin=-1.0, vmax=1.0)
    pl.title(r'Time = ' + "%.2f"%(time_array[file_number]) + " ps")

    pl.streamplot(q1, q2, 
                  combined_vel_drift_x_array, combined_vel_drift_y_array,
                  density=2, color='black',
                  linewidth=0.7, arrowsize=1
                 )

    pl.colorbar(im)
    
    pl.xlim([domain_1.q1_start, domain_1.q1_end])
    pl.ylim([domain_1.q2_start, domain_1.q2_end])

    pl.gca().set_xticks(q1 - 0.5*dq1)
    pl.gca().set_yticks(q2 - 0.5*dq2)

    pl.grid(True)
    pl.plot(q1_meshgrid, q2_meshgrid, marker='.', markersize = 2, color='k', linestyle='none')

    pl.gca().xaxis.set_ticklabels([])
    pl.gca().yaxis.set_ticklabels([])

    if (params_1.horizontal_internal_bcs_enabled):
        mirror_indices = q1_1[((q1_1 >= params_1.horizontal_mirror_0_start) & \
                            (q1_1 <= params_1.horizontal_mirror_0_end))]


        pl.axhline(0.25, color = 'k', ls = '--')
        pl.axhspan(ymin = q2_1[params_1.horizontal_mirror_0_index - 2*N_g] - dq2_1/2,
                   ymax = q2_1[params_1.horizontal_mirror_0_index] - dq2_1/2,
                   xmin = (mirror_indices[0]-dq1_1/2)/domain_1.q1_end,
                   xmax = (mirror_indices[-1]+dq1_1/2)/domain_1.q1_end,
                   color='k', alpha = 0.5)
        
    
    if (params_1.vertical_internal_bcs_enabled):
        mirror_indices = q2_1[((q2_1 >= params_1.vertical_mirror_0_start) & \
                            (q2_1 <= params_1.vertical_mirror_0_end))]

        pl.axvline(0.75, color = 'k', ls = '--')
        pl.axvspan(xmin = q1_1[params_1.vertical_mirror_0_index - 2*N_g] - dq1_1/2,
                   xmax = q1_1[params_1.vertical_mirror_0_index] - dq1_1/2,
                   ymin = (mirror_indices[0]-dq2_1/2)/domain.q2_end,
                   ymax = (mirror_indices[-1]+dq2_1/2)/domain.q2_end,
                   color='k', alpha = 0.5)
        

    
    pl.gca().set_aspect('equal')
    pl.xlabel(r'$x\;(\mu \mathrm{m})$')
    pl.ylabel(r'$y\;(\mu \mathrm{m})$')


    pl.suptitle('$\\tau_\mathrm{mc} = \infty$ ps, $\\tau_\mathrm{mr} = 1000.0$ ps')
    pl.savefig('images/dump_' + '%06d'%file_number + '.png')
    pl.clf()
    
