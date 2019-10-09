import arrayfire as af
import numpy as np
from scipy.signal import correlate
from scipy.interpolate import interp2d
import scipy.fftpack
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

N_p1 = domain.N_p1
N_p2 = domain.N_p2
N_samples = N_p2
step_size = (domain.p2_end - domain.p2_start)/N_p2
print ('Step size :', step_size)

p2 = domain.p2_start + (0.5 + np.arange(N_p2)) * (domain.p2_end - domain.p2_start)/N_p2
    
dt = params.dt
dump_interval = params.dump_steps
    
PSD_vs_tau_mr = []
tau_mr = [100.0]
for index in tau_mr:
    filepath = \
    '/home/mchandra/gitansh/gitansh_bolt/example_problems/electronic_boltzmann/L_1.0_2.5_tau_ee_inf_tau_eph_%.2f_DC/dumps'%index
    distribution_function_files = np.sort(glob.glob(filepath+'/f_*.h5'))
    
    time_array = np.loadtxt(filepath+"/../dump_time_array.txt")
    
    h5f  = h5py.File(distribution_function_files[0], 'r')
    dist_func_background = h5f['distribution_function'][:]
    h5f.close()
    
    h5f  = h5py.File(distribution_function_files[-1], 'r')
    dist_func = h5f['distribution_function'][:]
    h5f.close()
    

    avg_PSD_array = []
    N_1 = 7
    N_2 = 7
    for index_1 in range(N_1):
        for index_2 in range(N_2):
    
            print('Index : ', index_1, index_2)
    
            q1_position = int(domain.N_q1*((index_1/N_1)+(1/(2*N_1))))
            q2_position = int(domain.N_q2*((index_2/N_2)+(1/(2*N_2))))
    #        
            a = np.max((dist_func - dist_func_background)[q2_position, q1_position, :])
            b = np.abs(np.min((dist_func - dist_func_background)[q2_position, q1_position, :]))
            norm_factor = 1.0#np.maximum(a, b)
            f_at_desired_q = \
            np.reshape((dist_func-dist_func_background)[q2_position, q1_position,:], [N_p2])/norm_factor
    
            avg_PSD_array.append(f_at_desired_q)
    
    avg_PSD_array = np.array(avg_PSD_array)
    PSD_vs_tau_mr.append(avg_PSD_array)

colors = ['C0', 'C1', 'C2', 'C3']
color_index = 0
shift_factor = 1.
for PSD in PSD_vs_tau_mr:    
    for data in PSD:
        xf = 2*np.pi*np.linspace(0.0, 1.0/(2.0*step_size), N_samples/2)
        yf = scipy.fftpack.fft(data)
    
        pl.loglog(xf, shift_factor * 2.0/N_samples * np.abs(yf[:int(N_samples/2)]), \
                alpha = 0.01, linewidth = 1, color = colors[color_index])

        #pl.loglog(xf[10:], xf[10:]**(-1.4), linestyle='--', color='black')
        #pl.xlim(xmax = 200)
        #pl.ylim(ymin=1e-4)

    color_index = color_index + 1
    shift_factor = shift_factor*1e-3

c0_patch = patches.Patch(color='C0', label='$\mathrm{\\tau_{mr}=0.5\ ps}$')
c1_patch = patches.Patch(color='C1', label='$\mathrm{\\tau_{mr}=0.6\ ps}$')
c2_patch = patches.Patch(color='C2', label='$\mathrm{\\tau_{mr}=0.7\ ps}$')
c3_patch = patches.Patch(color='C3', label='$\mathrm{\\tau_{mr}=100.0\ ps}$')
#c2_patch = patches.Patch(color='C2', label='$\mathrm{\\tau_{mr}=100.0 ps}$')

#pl.legend(handles=[c0_patch, c1_patch, c2_patch, c3_patch], loc=3)

#leg = pl.gca().get_legend()
#leg.legendHandles[0].set_color('C0')
#leg.legendHandles[1].set_color('C1')
#leg.legendHandles[2].set_color('C2')
#leg.legendHandles[3].set_color('C3')

pl.xlabel ('$\mathrm{k_{\\theta}}$')
pl.ylabel ('$\mathrm{\hat{f}(k_{\\theta})}$')
    
pl.tight_layout()
pl.savefig('images/f_vs_theta_combined.png')
pl.clf()

