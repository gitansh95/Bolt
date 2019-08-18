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
sensor_1_left_start = source_end # um
sensor_1_left_end   = domain.q2_end # um

sensor_1_right_start = drain_end # um
sensor_1_right_end   = domain.q2_end # um

sensor_1_left_indices  = (q2 > sensor_1_left_start ) & (q2 < sensor_1_left_end)
sensor_1_right_indices = (q2 > sensor_1_right_start) & (q2 < sensor_1_right_end)

tau = np.arange(0.1, 0.91, 0.1)

for index in tau:
    filepath = \
    '/home/mchandra/gitansh/gitansh_bolt/example_problems/electronic_boltzmann/L_0.500_1.250_tau_ee_inf_tau_eph_%.2f/dumps'%index
    moment_files 		  = np.sort(glob.glob(filepath+'/moment*.h5'))
    lagrange_multiplier_files = \
        np.sort(glob.glob(filepath+'/lagrange_multipliers*.h5'))

    time_array = np.loadtxt(filepath + '/../dump_time_array.txt')

    dt = params.dt
    dump_interval = params.dump_steps

    sensor_1_signal_array = []
    print("Loading data...")
    density = []
    edge_density = []
    for file_number, dump_file in enumerate(moment_files):

        print("File number = ", file_number, ' of ', moment_files.size)
        h5f  = h5py.File(dump_file, 'r')
        moments = np.swapaxes(h5f['moments'][:], 0, 1)
        h5f.close()

        density.append(moments[:, :, 0])
        edge_density.append(density[file_number][0, sensor_1_left_indices])

    density = np.array(density)
    edge_density = np.array(edge_density)
    
    mean_density = np.mean(density)
    max_density  = np.max(density)
    min_density  = np.min(density)
    
    np.savetxt("data/edge_density_L_0.500_1.250_tau_ee_inf_tau_eph_%.2f.txt"%index, \
            edge_density - mean_density)
    np.savetxt("data/q2_edge_L_0.500_1.250_tau_ee_inf_tau_eph_%.2f.txt"%index, q2[sensor_1_left_indices])
    np.savetxt("data/time_L_0.500_1.250_tau_ee_inf_tau_eph_%.2f.txt"%index, time_array)


#print("Dumping data...")
#for file_number in yt.parallel_objects(range(density.shape[0])):

    #print("File number = ", file_number, ' of ', moment_files.size)
    
    #pl.semilogy(q2[sensor_1_left_indices], 
    #            density[file_number][0, sensor_1_left_indices],
    #           )
    #pl.title(r'Time = ' + "%.2f"%(time_array[file_number]) + " ps")
    #pl.title(r'Time = ' + "%.2f"%(file_number*dt*dump_interval) + " ps")

    #pl.xlim([sensor_1_left_start, sensor_1_left_end])
    #pl.ylim([min_density-mean_density, max_density-mean_density])
    #pl.ylim([0., np.log(max_density)])
    
    #pl.gca().set_aspect('equal')
    #pl.xlabel(r'$x\;(\mu \mathrm{m})$')
    #pl.ylabel(r'$y\;(\mu \mathrm{m})$')
    
    #pl.suptitle('$\\tau_\mathrm{mc} = \infty$ ps, $\\tau_\mathrm{mr} = 3.0$ ps')
    #pl.savefig('images/dump_' + '%06d'%file_number + '.png')
    #pl.savefig('images/density_' + '%06d'%file_number + '.png')
    #pl.clf()
    
    

