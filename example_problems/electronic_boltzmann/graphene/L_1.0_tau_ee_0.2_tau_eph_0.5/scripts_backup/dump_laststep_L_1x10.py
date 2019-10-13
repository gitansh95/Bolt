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


# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 25, 7.5
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

import PetscBinaryIO
io = PetscBinaryIO.PetscBinaryIO()

N_q1 = 72
q1_start = 0.0
q1_end = 1.0
q1 = q1_start + (0.5 + np.arange(N_q1)) * (q1_end - q1_start)/N_q1

source_start = 0.0
source_end   = 0.25

drain_start  = 0.0
drain_end    = 0.25

dt = 0.025/4
dump_interval = 5


heights = np.arange(0.5, 0.71, 0.10)

for index in heights:
    print ('Index : ', index)

    filepath = \
    '/home/mchandra/cci_data/dumps_L_1.0_20.0_tau_eph_%.1f'%index
    moment_files 		  = np.sort(glob.glob(filepath+'/moment*.h5'))
    lagrange_multiplier_files = \
            np.sort(glob.glob(filepath+'/lagrange_multipliers*.h5'))

    file_number = -2
    print (filepath)
    print("file number = ", file_number, "of ", moment_files.size)
    
    q2_start = 0.0
    q2_end = 10.0
    N_q2 = int(round(q2_end*N_q1))
    print ('N_q2 : ', N_q2)
    q2 = q2_start + (0.5 + np.arange(N_q2)) * (q2_end - q2_start)/N_q2
    print ('q2 : ', q2)

    q2_meshgrid, q1_meshgrid = np.meshgrid(q2, q1)

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
    #j_x_2 = density*vel_drift_x
    #j_y_2 = density*vel_drift_y

    from matplotlib import transforms, colors
    base = pl.gca().transData
    rot = transforms.Affine2D().rotate_deg(-90)
  
    #print("file_number = ", file_number, "vel_drift_x.shape = ", vel_drift_x.shape)
    #print("file_number = ", file_number, "vel_drift_y.shape = ", vel_drift_y.shape)

    #print (density)
    
    delta_n = density - np.mean(density)
   
    #pl.subplot(2,1,1)

    im = pl.contourf(q1, q2, delta_n, 200,
                norm=colors.SymLogNorm(linthresh=delta_n.max()/20), cmap='bwr',
                    transform = rot + base)
    #pl.contourf(q1, q2, delta_n, 200, cmap = 'bwr', transform = rot + base)

    pl.streamplot(q1, q2, j_x_2, j_y_2,
                  density=q2_end*1.5, color='black',
                  linewidth=0.7, arrowsize=1, transform = rot +
                   base
                 )

    pl.xlim([0, q2_end])
    pl.ylim([-1, 0])
    #pl.xlim([q1_start, q1_end])
    #pl.ylim([q2_start, q2_end])
    
    pl.gca().set_aspect('equal')
    pl.xlabel(r'$x\;(\mu \mathrm{m})$')
    pl.ylabel(r'$y\;(\mu \mathrm{m})$')

    #pl.colorbar(im)
    
    #pl.tight_layout()

    #pl.suptitle('$\\tau_\mathrm{mc} = \infty$ ps, $\\tau_\mathrm{mr} = \infty$ ps')
    pl.savefig('images/dump_laststep_L_1.0_20.0_tau_eph_%.2f.png'%index)
    pl.clf()
    
