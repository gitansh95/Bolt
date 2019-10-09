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


# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 20, 7.5
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

import PetscBinaryIO
io = PetscBinaryIO.PetscBinaryIO()

N_q1 = 72
q1_start = 0.0
q1_end = 1.0
q1 = q1_start + (0.5 + np.arange(N_q1)) * (q1_end - q1_start)/N_q1
    
q2_start  = 0.0
q2_end = 10.0
N_q2_full = int(round(10.0*N_q1))
q2_full   = q2_start + (0.5 + np.arange(N_q2_full)) * (q2_end - q2_start)/N_q2_full

source_start = 0.0
source_end   = 0.25

drain_start  = 0.0
drain_end    = 0.25

dt = 0.025/4
dump_interval = 5

####

heights_1 = np.arange(0.25, 5.01, 0.05)
heights_2 = np.arange(5.25, 10.001, 0.25)

heights   = heights_1
heights = np.append(heights, heights_2)

local_conductivity_array = []
nonlocal_conductivity_array = []

dump_step = dump_interval
time_step = dt

voltage_local_list_tau_mr_inf_vel_1em3    = []
voltage_nonlocal_list_tau_mr_inf_vel_1em3 = []
time_list_tau_mr_inf_vel_1em3             = []
drive_current_tau_mr_inf_vel_1em3         = []
q2_tau_mr_inf_vel_1em3                    = []
resistance_local_list_tau_mr_inf_vel_1em3 = []

####
for index in heights:
    print ('Index : ', index)

    q2_start = 0.0
    q2_end   = index
    N_q2     = int(round(q2_end*72))
    q2       = q2_start + (0.5 + np.arange(N_q2)) * (q2_end - q2_start)/N_q2

    q2_tau_mr_inf_vel_1em3.append(q2)

    local_indices = q2 < 0.25
    nonlocal_indices = (q2 > .25) & (q2 < .5)

    voltages    = \
        np.load('edge_data/edge_L_1.0_%.2f_tau_ee_inf_tau_eph_inf.txt.npz'%index)
    voltage_left = voltages['left']
    voltage_right = voltages['right']

    currents = \
        np.loadtxt('edge_data/current_L_1.0_%.2f_tau_ee_inf_tau_eph_inf.txt'%index)
    drive_current = np.mean(currents[local_indices])
    drive_current_tau_mr_inf_vel_1em3.append(drive_current)

    time = dump_step * time_step * np.arange(0, voltage_left[:, 0].size, 1)
    indices = np.where(time > time[-1]- 100)
    time_list_tau_mr_inf_vel_1em3.append(time[indices])

    left_probe = np.mean(voltage_left[:, local_indices], axis=1)
    right_probe = np.mean(voltage_right[:, local_indices], axis=1)
    voltage_local =  left_probe - right_probe

    left_probe = np.mean(voltage_left[:, nonlocal_indices], axis=1)
    right_probe = np.mean(voltage_right[:, nonlocal_indices], axis=1)
    voltage_nonlocal =  left_probe - right_probe

    voltage_local_list_tau_mr_inf_vel_1em3.append(voltage_local[indices])
    voltage_nonlocal_list_tau_mr_inf_vel_1em3.append(voltage_nonlocal[indices])

    local_resistance = np.mean(voltage_local)/drive_current
    resistance_local_list_tau_mr_inf_vel_1em3.append(local_resistance)


heights_2 = heights*2


for index in heights_2:
    print ('Index : ', index)

    filepath = \
    '/home/mchandra/cci_data/dumps_ballistic_tau_inf_L_1.0_%.1f'%index
    moment_files 		  = np.sort(glob.glob(filepath+'/moment*.h5'))
    lagrange_multiplier_files = \
            np.sort(glob.glob(filepath+'/lagrange_multipliers*.h5'))

    file_number = -2
    print (filepath)
    print("file number = ", file_number, "of ", moment_files.size)
    
    q2_start = 0.0
    q2_end = index/2
    N_q2 = int(round(q2_end*N_q1))
    print ('N_q2 : ', N_q2)
    q2 = q2_start + (0.5 + np.arange(N_q2)) * (q2_end - q2_start)/N_q2
    #print ('q2 : ', q2)

    q2_meshgrid, q1_meshgrid = np.meshgrid(q2_full, q1)

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
    j_x_2 = lagrange_multipliers[:, :, 5]
    j_y_2 = lagrange_multipliers[:, :, 6]

    
    delta_n = density - np.mean(density)

    N_cols = 1#int(round(4/q2_end))
    gs = gridspec.GridSpec(6, N_cols)

   
    ax1 = pl.subplot(gs[:4, 0])
    
    density_full_array = np.zeros((N_q2_full, N_q1))
    j_x_2_full_array   = np.zeros((N_q2_full, N_q1))
    j_y_2_full_array   = np.zeros((N_q2_full, N_q1))

    density_full_array[:N_q2, :] = delta_n
    j_x_2_full_array[:N_q2, :]   = j_x_2
    j_y_2_full_array[:N_q2, :]   = j_y_2
    
    density_min = -11000#np.min(density_full_array)
    density_max = 11000#np.max(density_full_array)

    print ('density_full_array.shape', density_full_array.shape)
    print ('q1.shape', q1.shape)
    print ('q2.shape', q2.shape)
    print ('q2_full.shape', q2_full.shape)
    
    from matplotlib import transforms, colors
    #base = pl.gca().transData
    base = ax1.transData
    rot = transforms.Affine2D().rotate_deg(-90)
  

    #im = ax1.contourf(q1, q2_full, density_full_array, 200,
    #            norm=colors.SymLogNorm(linthresh=density_full_array.max()/20), cmap='bwr',
    #                transform = rot + base)
    
    im = ax1.contourf(q1, q2_full,
                density_full_array, 200,
                norm = MidpointNormalize(midpoint=0, vmin=density_min, vmax=density_max),
                cmap='bwr', transform = rot + base
                     )

    ax1.streamplot(q1, q2_full,
                j_x_2_full_array,
                j_y_2_full_array,
                density=15, color='black',
                linewidth=0.5, arrowsize=1, transform = rot + base
                  )

    #ax1.set_xlim([0, q2_end])
    ax1.set_xlim([0, 10.0])
    ax1.set_ylim([-1, 0])
    
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    #ax1.set_aspect('equal')
    ax1.set_aspect(2)
    #ax1.set_xlabel(r'$x\;(\mu \mathrm{m})$')
    #ax1.set_ylabel(r'$y\;(\mu \mathrm{m})$')

    #pl.colorbar(im)

    ax2 = pl.subplot(gs[4:, 0])
    
    norm = 1.001*resistance_local_list_tau_mr_inf_vel_1em3[-1]
    ax2.plot(heights, \
        (resistance_local_list_tau_mr_inf_vel_1em3/norm), \
        marker = 'o', markersize=3, lw=2, label="$\mathrm{\\tau_{mr} = \infty,\ vel = 1e^{-3} }$")
    
    #pl.axhline(1, color='C0', linestyle='--', lw=3)
    ax2.set_xlim(xmin=0.0, xmax=10.0)
    ax2.set_ylim(ymax=1.10)

    ax2.axvline(q2_end, color='k', ls='--')
    #pl.xlim([0.2, 3.5])

    
    #pl.tight_layout()
    #pl.subplots_adjust(left=0.0)

    #pl.suptitle('$\\tau_\mathrm{mc} = \infty$ ps, $\\tau_\mathrm{mr} = \infty$ ps')

    pl.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax = pl.axes([0.85, 0.1, 0.075, 0.8])
    pl.colorbar(im, cax=cax)

    pl.savefig('images/dump_tau_mr_inf_laststep_L_1.0_%.2f.png'%q2_end)
    pl.clf()
    
