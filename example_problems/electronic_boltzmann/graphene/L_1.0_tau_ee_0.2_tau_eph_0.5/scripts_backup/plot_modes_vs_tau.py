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
pl.rcParams['figure.figsize']  = 12, 7.5
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

N_q1 = 18
N_q2 = 45

q1_start = 0.
q1_end   = 1.0

q2_start = 0.
q2_end   = 2.5

q1 = q1_start + (0.5 + np.arange(N_q1)) * (q1_end - q1_start)/N_q1
q2 = q2_start + (0.5 + np.arange(N_q2)) * (q2_end - q2_start)/N_q2

q2_meshgrid, q1_meshgrid = np.meshgrid(q2, q1)

drive_start = 1.0
drive_end = 1.5


# tau_mc = inf
tau_mr_1 = np.arange(0.01, 0.191, 0.01)
tau_mr_2 = np.arange(0.2, 0.491, 0.01)
tau_mr_3 = np.arange(0.5, 10.01, 0.1)
tau_mr_4 = np.arange(11.0, 100.01, 1.0)

tau_mr = tau_mr_1
tau_mr = np.append(tau_mr, tau_mr_2)
tau_mr = np.append(tau_mr, tau_mr_3)
tau_mr = np.append(tau_mr, tau_mr_4)

# tau_mr = 100
tau_mc_1 = np.arange(0.01, 0.191, 0.01)
tau_mc_2 = np.arange(0.2, 0.501, 0.01)
tau_mc_3 = np.arange(0.6, 10.01, 0.1)
tau_mc_4 = np.arange(11.0, 99.01, 1.0)

tau_mc = tau_mc_1
tau_mc = np.append(tau_mc, tau_mc_2)
tau_mc = np.append(tau_mc, tau_mc_3)
tau_mc = np.append(tau_mc, tau_mc_4)

change_N_at = tau_mc_1.size #used to change N_p2 as required

# tau_mc = 0.01
tau_mr_1 = np.arange(0.01, 0.191, 0.01)
tau_mr_2 = np.arange(0.20, 0.501, 0.01)
tau_mr_3 = np.arange(0.6, 10.01, 0.1)
tau_mr_4 = np.arange(11.0, 99.01, 1.0)

tau_mr_first_order = tau_mr_1
tau_mr_first_order = np.append(tau_mr_first_order, tau_mr_2)
tau_mr_first_order = np.append(tau_mr_first_order, tau_mr_3)
tau_mr_first_order = np.append(tau_mr_first_order, tau_mr_4)

plot_breaks = [tau_mr.size, tau_mr.size + tau_mc.size]

p2_start = -np.pi
p2_end   =  np.pi

tau_array = np.append(tau_mr, np.flip(tau_mc))
tau_array = np.append(tau_array, tau_mr_first_order)
    
gs  = gridspec.GridSpec(6, 6)
ax2 = pl.subplot(gs[:, :2])
ax3 = pl.subplot(gs[:, 2:4])
ax4 = pl.subplot(gs[:, 4:6])
    
ax2.set_xlim([0.01, 100.])
ax2.set_ylim([1e-7, 1e-1])
ax2.set_xlabel('$\\tau_{\mathrm{mr}}$')
ax2.set_ylabel('$\mathrm{Mode\ Amplitude}$', fontsize=16)

ax3.axvline(100, color = 'k', alpha = 0.5, linestyle = '--' , linewidth=1)

ax3.set_xlim([0.01, 100.])
ax3.set_ylim([1e-7, 1e-1])
ax3.invert_xaxis()
ax3.set_xlabel('$\\tau_{\mathrm{mc}}$')

ax4.set_xlim([0.01, 100.])
ax4.set_ylim([1e-7, 1e-1])
ax4.invert_xaxis()
ax4.set_xlabel('$\\tau_{\mathrm{mr}}$')
ax4.legend(bbox_to_anchor=(1.0, 1),prop={'size':13})

avg_mode_zero_ohmic = []
avg_mode_one_ohmic = []
avg_mode_others_ohmic = []
avg_mode_zero_hydro = []
avg_mode_one_hydro = []
avg_mode_others_hydro = []
avg_mode_zero_first_order = []
avg_mode_one_first_order = []
avg_mode_others_first_order = []

avg_mode_zero_ohmic_2 = []
avg_mode_one_ohmic_2 = []
avg_mode_others_ohmic_2 = []
avg_mode_zero_hydro_2 = []
avg_mode_one_hydro_2 = []
avg_mode_others_hydro_2 = []
avg_mode_zero_first_order_2 = []
avg_mode_one_first_order_2 = []
avg_mode_others_first_order_2 = []

N = 7
for index_1 in range(N):
  for index_2 in range(N):
    print (index_1, index_2)
    
    q1_position = q1[int(N_q1*((index_1/N)+(1/(2*N))))]
    q2_position = q2[int(N_q2*((index_2/N)+(1/(2*N))))]
    
    print ('q2 position : ', q2_position)
    if (q2_position > drive_end):
        print ("Adding plot")

        mode_zero_array_ohmic = \
            np.loadtxt('ohmic_ballistic/mode_zero_L_1.0_2.5_tau_ee_inf_%d_%d.txt'%(index_1, index_2))
        mode_one_array_ohmic = \
            np.loadtxt('ohmic_ballistic/mode_one_L_1.0_2.5_tau_ee_inf_%d_%d.txt'%(index_1, index_2))
        mode_others_array_ohmic = \
            np.loadtxt('ohmic_ballistic/mode_others_L_1.0_2.5_tau_ee_inf_%d_%d.txt'%(index_1, index_2))
    
        mode_zero_array_hydro = \
            np.loadtxt('hydro_ballistic/mode_zero_L_1.0_2.5_tau_eph_100.0_%d_%d.txt'%(index_1, index_2))
        mode_one_array_hydro = \
            np.loadtxt('hydro_ballistic/mode_one_L_1.0_2.5_tau_eph_100.0_%d_%d.txt'%(index_1, index_2))
        mode_others_array_hydro = \
            np.loadtxt('hydro_ballistic/mode_others_L_1.0_2.5_tau_eph_100.0_%d_%d.txt'%(index_1, index_2))
    
        mode_zero_array_first_order = \
            np.loadtxt('ohmic_hydro/mode_zero_L_1.0_2.5_tau_ee_0.01_%d_%d.txt'%(index_1, index_2))
        mode_one_array_first_order = \
            np.loadtxt('ohmic_hydro/mode_one_L_1.0_2.5_tau_ee_0.01_%d_%d.txt'%(index_1, index_2))
        mode_others_array_first_order = \
            np.loadtxt('ohmic_hydro/mode_others_L_1.0_2.5_tau_ee_0.01_%d_%d.txt'%(index_1, index_2))
    
    
        #ax2.loglog((tau_mr), mode_zero_array_ohmic, '-', label = '$A$', lw=2,
        #        alpha = 0.1, color = 'C0')
        #ax2.loglog((tau_mr), mode_one_array_ohmic, '-', label = '$B$', lw=2,
        #        alpha = 0.1, color = 'C1')
        #ax2.loglog((tau_mr), mode_others_array_ohmic, '-', label='Fluctuations', lw=2,
        #        alpha = 0.1, color = 'C2')
        ax2.set_xticks([0.01, 1])
        #ax2.set_yticks([])
        ax2.spines['right'].set_visible(False)
    
        #ax3.loglog((tau_mc), mode_zero_array_hydro, label = '$A$', lw=2,
        #        alpha = 0.1, color = 'C0')
        #ax3.loglog((tau_mc), mode_one_array_hydro, label = '$B$', lw=2,
        #        alpha = 0.1, color = 'C1')
        #ax3.loglog((tau_mc), mode_others_array_hydro, label='Fluctuations', lw=2,
        #        alpha = 0.1, color = 'C2')
        ax3.set_xticks([0.01, 1, 100])
        ax3.set_yticks([])
        ax3.spines['left'].set_visible(False)
        ax3.spines['right'].set_visible(False)
    
        #ax4.loglog((tau_mr_first_order), mode_zero_array_first_order, '-', label = '$A$', lw=2,
        #        alpha = 0.1, color = 'C0')
        #ax4.loglog((tau_mr_first_order), mode_one_array_first_order, '-', label = '$B$', lw=2,
        #        alpha = 0.1, color = 'C1')
        #ax4.loglog((tau_mr_first_order), mode_others_array_first_order, '-', label='Fluctuations', lw=2,
        #        alpha = 0.1, color = 'C2')
        ax4.axvline(100, color = 'k', alpha = 0.5, linestyle = '--', linewidth=1 )
        ax4.set_xticks([0.01, 1])
        ax4.set_yticks([])
        ax4.spines['left'].set_visible(False)

        avg_mode_zero_ohmic.append(mode_zero_array_ohmic)
        avg_mode_one_ohmic.append(mode_one_array_ohmic)
        avg_mode_others_ohmic.append(mode_others_array_ohmic)
        avg_mode_zero_hydro.append(mode_zero_array_hydro)
        avg_mode_one_hydro.append(mode_one_array_hydro)
        avg_mode_others_hydro.append(mode_others_array_hydro)
        avg_mode_zero_first_order.append(mode_zero_array_first_order)
        avg_mode_one_first_order.append(mode_one_array_first_order)
        avg_mode_others_first_order.append(mode_others_array_first_order)
    
    if (q2_position > drive_start):
        print ("Adding plot")

        mode_zero_array_ohmic = \
            np.loadtxt('ohmic_ballistic/mode_zero_L_1.0_2.5_tau_ee_inf_%d_%d.txt'%(index_1, index_2))
        mode_one_array_ohmic = \
            np.loadtxt('ohmic_ballistic/mode_one_L_1.0_2.5_tau_ee_inf_%d_%d.txt'%(index_1, index_2))
        mode_others_array_ohmic = \
            np.loadtxt('ohmic_ballistic/mode_others_L_1.0_2.5_tau_ee_inf_%d_%d.txt'%(index_1, index_2))
    
        mode_zero_array_hydro = \
            np.loadtxt('hydro_ballistic/mode_zero_L_1.0_2.5_tau_eph_100.0_%d_%d.txt'%(index_1, index_2))
        mode_one_array_hydro = \
            np.loadtxt('hydro_ballistic/mode_one_L_1.0_2.5_tau_eph_100.0_%d_%d.txt'%(index_1, index_2))
        mode_others_array_hydro = \
            np.loadtxt('hydro_ballistic/mode_others_L_1.0_2.5_tau_eph_100.0_%d_%d.txt'%(index_1, index_2))
    
        mode_zero_array_first_order = \
            np.loadtxt('ohmic_hydro/mode_zero_L_1.0_2.5_tau_ee_0.01_%d_%d.txt'%(index_1, index_2))
        mode_one_array_first_order = \
            np.loadtxt('ohmic_hydro/mode_one_L_1.0_2.5_tau_ee_0.01_%d_%d.txt'%(index_1, index_2))
        mode_others_array_first_order = \
            np.loadtxt('ohmic_hydro/mode_others_L_1.0_2.5_tau_ee_0.01_%d_%d.txt'%(index_1, index_2))
        
        avg_mode_zero_ohmic_2.append(mode_zero_array_ohmic)
        avg_mode_one_ohmic_2.append(mode_one_array_ohmic)
        avg_mode_others_ohmic_2.append(mode_others_array_ohmic)
        avg_mode_zero_hydro_2.append(mode_zero_array_hydro)
        avg_mode_one_hydro_2.append(mode_one_array_hydro)
        avg_mode_others_hydro_2.append(mode_others_array_hydro)
        avg_mode_zero_first_order_2.append(mode_zero_array_first_order)
        avg_mode_one_first_order_2.append(mode_one_array_first_order)
        avg_mode_others_first_order_2.append(mode_others_array_first_order)
        
avg_mode_zero_ohmic = np.array(avg_mode_zero_ohmic)
avg_mode_one_ohmic = np.array(avg_mode_one_ohmic)
avg_mode_others_ohmic = np.array(avg_mode_others_ohmic)
avg_mode_zero_hydro = np.array(avg_mode_zero_hydro)
avg_mode_one_hydro = np.array(avg_mode_one_hydro)
avg_mode_others_hydro = np.array(avg_mode_others_hydro)
avg_mode_zero_first_order = np.array(avg_mode_zero_first_order)
avg_mode_one_first_order = np.array(avg_mode_one_first_order)
avg_mode_others_first_order = np.array(avg_mode_others_first_order)

avg_mode_zero_ohmic = np.mean(avg_mode_zero_ohmic, axis=0)
avg_mode_one_ohmic = np.mean(avg_mode_one_ohmic, axis=0)
avg_mode_others_ohmic = np.mean(avg_mode_others_ohmic, axis=0)
avg_mode_zero_hydro = np.mean(avg_mode_zero_hydro, axis=0)
avg_mode_one_hydro = np.mean(avg_mode_one_hydro, axis=0)
avg_mode_others_hydro = np.mean(avg_mode_others_hydro, axis=0)
avg_mode_zero_first_order = np.mean(avg_mode_zero_first_order, axis=0)
avg_mode_one_first_order = np.mean(avg_mode_one_first_order, axis=0)
avg_mode_others_first_order = np.mean(avg_mode_others_first_order, axis=0)

avg_mode_zero_ohmic_2 = np.array(avg_mode_zero_ohmic_2)
avg_mode_one_ohmic_2 = np.array(avg_mode_one_ohmic_2)
avg_mode_others_ohmic_2 = np.array(avg_mode_others_ohmic_2)
avg_mode_zero_hydro_2 = np.array(avg_mode_zero_hydro_2)
avg_mode_one_hydro_2 = np.array(avg_mode_one_hydro_2)
avg_mode_others_hydro_2 = np.array(avg_mode_others_hydro_2)
avg_mode_zero_first_order_2 = np.array(avg_mode_zero_first_order_2)
avg_mode_one_first_order_2 = np.array(avg_mode_one_first_order_2)
avg_mode_others_first_order_2 = np.array(avg_mode_others_first_order_2)

avg_mode_zero_ohmic_2 = np.mean(avg_mode_zero_ohmic_2, axis=0)
avg_mode_one_ohmic_2 = np.mean(avg_mode_one_ohmic_2, axis=0)
avg_mode_others_ohmic_2 = np.mean(avg_mode_others_ohmic_2, axis=0)
avg_mode_zero_hydro_2 = np.mean(avg_mode_zero_hydro_2, axis=0)
avg_mode_one_hydro_2 = np.mean(avg_mode_one_hydro_2, axis=0)
avg_mode_others_hydro_2 = np.mean(avg_mode_others_hydro_2, axis=0)
avg_mode_zero_first_order_2 = np.mean(avg_mode_zero_first_order_2, axis=0)
avg_mode_one_first_order_2 = np.mean(avg_mode_one_first_order_2, axis=0)
avg_mode_others_first_order_2 = np.mean(avg_mode_others_first_order_2, axis=0)

#pl.tight_layout()
pl.subplots_adjust(wspace=0.02)
pl.subplots_adjust(hspace=-0.2)
        
ax2.loglog((tau_mr), avg_mode_zero_ohmic, '-', label = '$a$', lw=2,
                alpha = 1, color = 'C0', ls='-')
ax2.loglog((tau_mr), avg_mode_one_ohmic, '-', label = '$b$', lw=2,
                alpha = 1, color = 'C1', ls='-')
ax2.loglog((tau_mr), avg_mode_others_ohmic, '-', label='fluctuations', lw=2,
                alpha = 1, color = 'C2', ls='-')
ax2.set_xticks([0.01, 1])
#ax2.set_yticks([])
ax2.spines['right'].set_visible(False)
    
ax3.loglog((tau_mc), avg_mode_zero_hydro, label = '$a$', lw=2,
                alpha = 1, color = 'C0', ls='-')
ax3.loglog((tau_mc), avg_mode_one_hydro, label = '$b$', lw=2,
                alpha = 1, color = 'C1', ls='-')
ax3.loglog((tau_mc), avg_mode_others_hydro, label='fluctuations', lw=2,
                alpha = 1, color = 'C2', ls='-')
ax3.set_xticks([0.01, 1, 100])
ax3.set_yticks([])
ax3.spines['left'].set_visible(False)
ax3.spines['right'].set_visible(False)

ax4.loglog((tau_mr_first_order), avg_mode_zero_first_order, '-', label = '$a$', lw=2,
                alpha = 1, color = 'C0', ls='-')
ax4.loglog((tau_mr_first_order), avg_mode_one_first_order, '-', label = '$b$', lw=2,
                alpha = 1, color = 'C1', ls='-')
ax4.loglog((tau_mr_first_order), avg_mode_others_first_order, '-', label='fluctuations', lw=2,
                alpha = 1, color = 'C2', ls='-')
ax4.axvline(100, color = 'k', alpha = 0.5, linestyle = '--', linewidth=1 )
ax4.set_xticks([0.01, 1])
ax4.set_yticks([])
ax4.spines['left'].set_visible(False)


ax2.loglog((tau_mr), avg_mode_zero_ohmic_2, '-', label = '$a$', lw=2,
                alpha = 1, color = 'C0', ls='--')
ax2.loglog((tau_mr), avg_mode_one_ohmic_2, '-', label = '$b$', lw=2,
                alpha = 1, color = 'C1', ls='--')
ax2.loglog((tau_mr), avg_mode_others_ohmic_2, '-', label='fluctuations', lw=2,
                alpha = 1, color = 'C2', ls='--')
ax2.set_xticks([0.01, 1])
#ax2.set_yticks([])
ax2.spines['right'].set_visible(False)
    
ax3.loglog((tau_mc), avg_mode_zero_hydro_2, label = '$a$', lw=2,
                alpha = 1, color = 'C0', ls='--')
ax3.loglog((tau_mc), avg_mode_one_hydro_2, label = '$b$', lw=2,
                alpha = 1, color = 'C1', ls='--')
ax3.loglog((tau_mc), avg_mode_others_hydro_2, label='fluctuations', lw=2,
                alpha = 1, color = 'C2', ls='--')
ax3.set_xticks([0.01, 1, 100])
ax3.set_yticks([])
ax3.spines['left'].set_visible(False)
ax3.spines['right'].set_visible(False)

ax4.loglog((tau_mr_first_order), avg_mode_zero_first_order_2, '-', label = '$a$', lw=2,
                alpha = 1, color = 'C0', ls='--')
ax4.loglog((tau_mr_first_order), avg_mode_one_first_order_2, '-', label = '$b$', lw=2,
                alpha = 1, color = 'C1', ls='--')
ax4.loglog((tau_mr_first_order), avg_mode_others_first_order_2, '-', label='fluctuations', lw=2,
                alpha = 1, color = 'C2', ls='--')
ax4.axvline(100, color = 'k', alpha = 0.5, linestyle = '--', linewidth=1 )
ax4.set_xticks([0.01, 1])
ax4.set_yticks([])
ax4.spines['left'].set_visible(False)

pl.savefig('images/mode_vs_tau.png', bbox_inches = 'tight',\
            pad_inches=0.1)






