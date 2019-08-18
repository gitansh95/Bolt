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


#tau_mr = 0.01
filepath = \
'/home/mchandra/gitansh/gitansh_bolt/example_problems/electronic_boltzmann/L_1.0_2.5_tau_ee_inf_tau_eph_0.01_DC/dumps'
moment_files 		        = np.sort(glob.glob(filepath+'/moment*.h5'))
lagrange_multiplier_files   = \
        np.sort(glob.glob(filepath+'/lagrange_multipliers*.h5'))
distribution_function_files = np.sort(glob.glob(filepath+'/f_*.h5'))

dt = params.dt
dump_interval = params.dump_steps

time_array = np.loadtxt("dump_time_array.txt")

h5f  = h5py.File(distribution_function_files[0], 'r')
dist_func_background = h5f['distribution_function'][:]
h5f.close()

h5f  = h5py.File(distribution_function_files[-1], 'r')
dist_func = h5f['distribution_function'][:]
h5f.close()

file_number = moment_files.size-1

N_q1 = 36
N_q2 = 90

avg_PSD_array_001 = []
N_1 = 12
N_2 = 30
for index_1 in range(N_1):
    for index_2 in range(N_2):

        print('Index : ', index_1, index_2)

        q1_position = int(N_q1*((index_1/N_1)+(1/(2*N_1))))
        q2_position = int(N_q2*((index_2/N_2)+(1/(2*N_2))))
#        
        a = np.max((dist_func - dist_func_background)[q2_position, q1_position, :])
        b = np.abs(np.min((dist_func - dist_func_background)[q2_position, q1_position, :]))
        norm_factor = np.maximum(a, b)
        f_at_desired_q = \
        np.reshape((dist_func-dist_func_background)[q2_position, q1_position,:], [N_p2])/norm_factor
        
        avg_PSD_array_001.append(f_at_desired_q)

#tau_mr = 0.1
filepath = \
'/home/mchandra/gitansh/gitansh_bolt/example_problems/electronic_boltzmann/L_1.0_2.5_tau_ee_inf_tau_eph_0.1_DC/dumps'
moment_files 		        = np.sort(glob.glob(filepath+'/moment*.h5'))
lagrange_multiplier_files   = \
        np.sort(glob.glob(filepath+'/lagrange_multipliers*.h5'))
distribution_function_files = np.sort(glob.glob(filepath+'/f_*.h5'))

dt = params.dt
dump_interval = params.dump_steps

time_array = np.loadtxt("dump_time_array.txt")

h5f  = h5py.File(distribution_function_files[0], 'r')
dist_func_background = h5f['distribution_function'][:]
h5f.close()

h5f  = h5py.File(distribution_function_files[-1], 'r')
dist_func = h5f['distribution_function'][:]
h5f.close()

file_number = moment_files.size-1

N_q1 = 36
N_q2 = 90

avg_PSD_array_01 = []
N_1 = 12
N_2 = 30
for index_1 in range(N_1):
    for index_2 in range(N_2):

        print('Index : ', index_1, index_2)

        q1_position = int(N_q1*((index_1/N_1)+(1/(2*N_1))))
        q2_position = int(N_q2*((index_2/N_2)+(1/(2*N_2))))
#        
        a = np.max((dist_func - dist_func_background)[q2_position, q1_position, :])
        b = np.abs(np.min((dist_func - dist_func_background)[q2_position, q1_position, :]))
        norm_factor = np.maximum(a, b)
        f_at_desired_q = \
        np.reshape((dist_func-dist_func_background)[q2_position, q1_position,:], [N_p2])/norm_factor
        
        avg_PSD_array_01.append(f_at_desired_q)


#tau_mr = 1.0
filepath = \
'/home/mchandra/gitansh/gitansh_bolt/example_problems/electronic_boltzmann/L_1.0_2.5_tau_ee_inf_tau_eph_1.0_DC/dumps_backup_32_90_8192'
moment_files 		        = np.sort(glob.glob(filepath+'/moment*.h5'))
lagrange_multiplier_files   = \
        np.sort(glob.glob(filepath+'/lagrange_multipliers*.h5'))
distribution_function_files = np.sort(glob.glob(filepath+'/f_*.h5'))

dt = params.dt
dump_interval = params.dump_steps

time_array = np.loadtxt("dump_time_array.txt")

h5f  = h5py.File(distribution_function_files[0], 'r')
dist_func_background = h5f['distribution_function'][:]
h5f.close()

h5f  = h5py.File(distribution_function_files[-1], 'r')
dist_func = h5f['distribution_function'][:]
h5f.close()

file_number = moment_files.size-1

N_q1 = 36
N_q2 = 90

avg_PSD_array_1 = []
N_1 = 12
N_2 = 30
for index_1 in range(N_1):
    for index_2 in range(N_2):

        print('Index : ', index_1, index_2)

        q1_position = int(N_q1*((index_1/N_1)+(1/(2*N_1))))
        q2_position = int(N_q2*((index_2/N_2)+(1/(2*N_2))))
#        
        a = np.max((dist_func - dist_func_background)[q2_position, q1_position, :])
        b = np.abs(np.min((dist_func - dist_func_background)[q2_position, q1_position, :]))
        norm_factor = np.maximum(a, b)
        f_at_desired_q = \
        np.reshape((dist_func-dist_func_background)[q2_position, q1_position,:], [N_p2])/norm_factor
        
        avg_PSD_array_1.append(f_at_desired_q)


#tau_mr = 2.0
filepath = \
'/home/mchandra/gitansh/gitansh_bolt/example_problems/electronic_boltzmann/L_1.0_2.5_tau_ee_inf_tau_eph_2.0_DC/dumps'
moment_files 		        = np.sort(glob.glob(filepath+'/moment*.h5'))
lagrange_multiplier_files   = \
        np.sort(glob.glob(filepath+'/lagrange_multipliers*.h5'))
distribution_function_files = np.sort(glob.glob(filepath+'/f_*.h5'))

dt = params.dt
dump_interval = params.dump_steps

time_array = np.loadtxt("dump_time_array.txt")

h5f  = h5py.File(distribution_function_files[0], 'r')
dist_func_background = h5f['distribution_function'][:]
h5f.close()

h5f  = h5py.File(distribution_function_files[-1], 'r')
dist_func = h5f['distribution_function'][:]
h5f.close()

file_number = moment_files.size-1

N_q1 = 36
N_q2 = 90

avg_PSD_array_2 = []
N_1 = 12
N_2 = 30
for index_1 in range(N_1):
    for index_2 in range(N_2):

        print('Index : ', index_1, index_2)

        q1_position = int(N_q1*((index_1/N_1)+(1/(2*N_1))))
        q2_position = int(N_q2*((index_2/N_2)+(1/(2*N_2))))
#        
        a = np.max((dist_func - dist_func_background)[q2_position, q1_position, :])
        b = np.abs(np.min((dist_func - dist_func_background)[q2_position, q1_position, :]))
        norm_factor = np.maximum(a, b)
        f_at_desired_q = \
        np.reshape((dist_func-dist_func_background)[q2_position, q1_position,:], [N_p2])/norm_factor
        
        avg_PSD_array_2.append(f_at_desired_q)


#tau_mr = 3.0
filepath = \
'/home/mchandra/gitansh/gitansh_bolt/example_problems/electronic_boltzmann/L_1.0_2.5_tau_ee_inf_tau_eph_3.0_DC/dumps'
moment_files 		        = np.sort(glob.glob(filepath+'/moment*.h5'))
lagrange_multiplier_files   = \
        np.sort(glob.glob(filepath+'/lagrange_multipliers*.h5'))
distribution_function_files = np.sort(glob.glob(filepath+'/f_*.h5'))

dt = params.dt
dump_interval = params.dump_steps

time_array = np.loadtxt("dump_time_array.txt")

h5f  = h5py.File(distribution_function_files[0], 'r')
dist_func_background = h5f['distribution_function'][:]
h5f.close()

h5f  = h5py.File(distribution_function_files[-1], 'r')
dist_func = h5f['distribution_function'][:]
h5f.close()

file_number = moment_files.size-1

N_q1 = 36
N_q2 = 90

avg_PSD_array_3 = []
N_1 = 12
N_2 = 30
for index_1 in range(N_1):
    for index_2 in range(N_2):

        print('Index : ', index_1, index_2)

        q1_position = int(N_q1*((index_1/N_1)+(1/(2*N_1))))
        q2_position = int(N_q2*((index_2/N_2)+(1/(2*N_2))))
#        
        a = np.max((dist_func - dist_func_background)[q2_position, q1_position, :])
        b = np.abs(np.min((dist_func - dist_func_background)[q2_position, q1_position, :]))
        norm_factor = np.maximum(a, b)
        f_at_desired_q = \
        np.reshape((dist_func-dist_func_background)[q2_position, q1_position,:], [N_p2])/norm_factor
        
        avg_PSD_array_3.append(f_at_desired_q)


#tau_mr = 4.0
filepath = \
'/home/mchandra/gitansh/gitansh_bolt/example_problems/electronic_boltzmann/L_1.0_2.5_tau_ee_inf_tau_eph_4.0_DC/dumps'
moment_files 		        = np.sort(glob.glob(filepath+'/moment*.h5'))
lagrange_multiplier_files   = \
        np.sort(glob.glob(filepath+'/lagrange_multipliers*.h5'))
distribution_function_files = np.sort(glob.glob(filepath+'/f_*.h5'))

dt = params.dt
dump_interval = params.dump_steps

time_array = np.loadtxt("dump_time_array.txt")

h5f  = h5py.File(distribution_function_files[0], 'r')
dist_func_background = h5f['distribution_function'][:]
h5f.close()

h5f  = h5py.File(distribution_function_files[-1], 'r')
dist_func = h5f['distribution_function'][:]
h5f.close()

file_number = moment_files.size-1

N_q1 = 36
N_q2 = 90

avg_PSD_array_4 = []
N_1 = 12
N_2 = 30
for index_1 in range(N_1):
    for index_2 in range(N_2):

        print('Index : ', index_1, index_2)

        q1_position = int(N_q1*((index_1/N_1)+(1/(2*N_1))))
        q2_position = int(N_q2*((index_2/N_2)+(1/(2*N_2))))
#        
        a = np.max((dist_func - dist_func_background)[q2_position, q1_position, :])
        b = np.abs(np.min((dist_func - dist_func_background)[q2_position, q1_position, :]))
        norm_factor = np.maximum(a, b)
        f_at_desired_q = \
        np.reshape((dist_func-dist_func_background)[q2_position, q1_position,:], [N_p2])/norm_factor
        
        avg_PSD_array_4.append(f_at_desired_q)


#tau_mr = 5.0
filepath = \
'/home/mchandra/gitansh/gitansh_bolt/example_problems/electronic_boltzmann/L_1.0_2.5_tau_ee_inf_tau_eph_5.0_DC/backup_dumps_36_90_8192'
moment_files 		        = np.sort(glob.glob(filepath+'/moment*.h5'))
lagrange_multiplier_files   = \
        np.sort(glob.glob(filepath+'/lagrange_multipliers*.h5'))
distribution_function_files = np.sort(glob.glob(filepath+'/f_*.h5'))

dt = params.dt
dump_interval = params.dump_steps

time_array = np.loadtxt("dump_time_array.txt")

h5f  = h5py.File(distribution_function_files[0], 'r')
dist_func_background = h5f['distribution_function'][:]
h5f.close()

h5f  = h5py.File(distribution_function_files[-1], 'r')
dist_func = h5f['distribution_function'][:]
h5f.close()

file_number = moment_files.size-1

N_q1 = 36
N_q2 = 90

avg_PSD_array_5 = []
N_1 = 12
N_2 = 30
for index_1 in range(N_1):
    for index_2 in range(N_2):

        print('Index : ', index_1, index_2)

        q1_position = int(N_q1*((index_1/N_1)+(1/(2*N_1))))
        q2_position = int(N_q2*((index_2/N_2)+(1/(2*N_2))))
#        
        a = np.max((dist_func - dist_func_background)[q2_position, q1_position, :])
        b = np.abs(np.min((dist_func - dist_func_background)[q2_position, q1_position, :]))
        norm_factor = np.maximum(a, b)
        f_at_desired_q = \
        np.reshape((dist_func-dist_func_background)[q2_position, q1_position,:], [N_p2])/norm_factor
        
        avg_PSD_array_5.append(f_at_desired_q)

coefficient = 5
exponent = -2.7

#pl.subplot(5, 1, 1)
#for PSD in avg_PSD_array_001:
#    xf = 2*np.pi*np.linspace(0.0, 1.0/(2.0*step_size), N_samples/2)
#    yf = scipy.fftpack.fft(PSD)
#    
#    pl.loglog(xf, (1e6)*2.0/N_samples * np.abs(yf[:int(N_samples/2)]), alpha = 0.01, \
#            linewidth = 1, color = 'C5')

#pl.xlabel ('$\mathrm{k_{\\theta}}$')
#pl.ylabel ('$\mathrm{\hat{f}(k_{\\theta})}$')

#pl.loglog(xf[90:250], (1e6)*xf[90:250]**(exponent), linestyle='--', color='C5')
#pl.xlim(xmax = 200)
#pl.ylim(ymin=1e-4)

#pl.subplot(5, 1, 1)
for PSD in avg_PSD_array_1:
    xf = 2*np.pi*np.linspace(0.0, 1.0/(2.0*step_size), N_samples/2)
    yf = scipy.fftpack.fft(PSD)
    
    pl.loglog(xf, (1e3)*2.0/N_samples * np.abs(yf[:int(N_samples/2)]), alpha = 0.01, \
            linewidth = 1, color = 'C6')

#pl.xlabel ('$\mathrm{k_{\\theta}}$')
#pl.ylabel ('$\mathrm{\hat{f}(k_{\\theta})}$')

pl.loglog(xf[90:250], 30*coefficient*(1e3)*xf[90:250]**(exponent),
        linestyle='--', color='k')
#pl.xlim(xmax = 200)
#pl.ylim(ymin=1e-4)

#pl.subplot(5, 1, 1)
for PSD in avg_PSD_array_1:
    xf = 2*np.pi*np.linspace(0.0, 1.0/(2.0*step_size), N_samples/2)
    yf = scipy.fftpack.fft(PSD)
    
    pl.loglog(xf, 2.0/N_samples * np.abs(yf[:int(N_samples/2)]), alpha = 0.01, \
            linewidth = 1, color = 'C0')

#pl.xlabel ('$\mathrm{k_{\\theta}}$')
#pl.ylabel ('$\mathrm{\hat{f}(k_{\\theta})}$')

pl.loglog(xf[90:250], 30*coefficient*xf[90:250]**(exponent), linestyle='--',
        color='k')
#pl.xlim(xmax = 200)
#pl.ylim(ymin=1e-4)

#pl.subplot(5, 1, 2)
for PSD in avg_PSD_array_2:
    xf = 2*np.pi*np.linspace(0.0, 1.0/(2.0*step_size), N_samples/2)
    yf = scipy.fftpack.fft(PSD)
    
    pl.loglog(xf, (1e-3)*2.0/N_samples * np.abs(yf[:int(N_samples/2)]), alpha = 0.01, \
            linewidth = 1, color = 'C1')

#pl.xlabel ('$\mathrm{k_{\\theta}}$')
#pl.ylabel ('$\mathrm{\hat{f}(k_{\\theta})}$')
pl.loglog(xf[90:300], 150*coefficient*(1e-3)*xf[90:300]**(exponent),
        linestyle='--', color='k')

#pl.subplot(5, 1, 3)
for PSD in avg_PSD_array_3:
    xf = 2*np.pi*np.linspace(0.0, 1.0/(2.0*step_size), N_samples/2)
    yf = scipy.fftpack.fft(PSD)
    
    pl.loglog(xf,(1e-6)* 2.0/N_samples * np.abs(yf[:int(N_samples/2)]), alpha = 0.01, \
            linewidth = 1, color = 'C2')

#pl.xlabel ('$\mathrm{k_{\\theta}}$')
#pl.ylabel ('$\mathrm{\hat{f}(k_{\\theta})}$')
pl.loglog(xf[90:500], 250*coefficient*(1e-6)*xf[90:500]**(exponent),
        linestyle='--', color='k')

#pl.subplot(5, 1, 4)
for PSD in avg_PSD_array_4:
    xf = 2*np.pi*np.linspace(0.0, 1.0/(2.0*step_size), N_samples/2)
    yf = scipy.fftpack.fft(PSD)
    
    pl.loglog(xf, (1e-9)*2.0/N_samples * np.abs(yf[:int(N_samples/2)]), alpha = 0.01, \
            linewidth = 1, color = 'C3')

#pl.xlabel ('$\mathrm{k_{\\theta}}$')
#pl.ylabel ('$\mathrm{\hat{f}(k_{\\theta})}$')
pl.loglog(xf[90:750], 500*coefficient*(1e-9)*xf[90:750]**(exponent),
        linestyle='--', color='k')

#pl.subplot(5, 1, 5)
for PSD in avg_PSD_array_5:
    xf = 2*np.pi*np.linspace(0.0, 1.0/(2.0*step_size), N_samples/2)
    yf = scipy.fftpack.fft(PSD)
    
    pl.loglog(xf, (1e-12)*2.0/N_samples * np.abs(yf[:int(N_samples/2)]), alpha = 0.01, \
            linewidth = 1, color = 'C4')

#pl.xlabel ('$\mathrm{k_{\\theta}}$')
#pl.ylabel ('$\mathrm{\hat{f}(k_{\\theta})}$')
pl.loglog(xf[90:1000], 500*coefficient*(1e-12)*xf[90:1000]**(exponent),
        linestyle='--', color='k')

pl.tight_layout()
pl.savefig('f_vs_theta_overplot_combined.png')
pl.clf()

