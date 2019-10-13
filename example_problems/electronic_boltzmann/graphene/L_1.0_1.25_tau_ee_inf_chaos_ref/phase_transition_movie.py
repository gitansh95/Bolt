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
tau_mr_1 = np.arange(0.01, 0.101, 0.01)
tau_mr_2 = np.arange(0.2, 4.91, 0.1)
tau_mr_3 = np.arange(0.5, 10.01, 0.1)
tau_mr_4 = np.arange(11.0, 100.01, 1.0)

tau_mr_first_order = tau_mr_1
tau_mr_first_order = np.append(tau_mr_first_order, tau_mr_2)

plot_breaks = [tau_mr.size, tau_mr.size + tau_mc.size]

mode_zero_array_ohmic = np.loadtxt('mode_zero_data.txt')
mode_one_array_ohmic = np.loadtxt('mode_one_data.txt')
mode_others_array_ohmic = np.loadtxt('mode_others_data.txt')

mode_zero_array_hydro = np.loadtxt('mode_zero_data_hydro.txt')
mode_one_array_hydro = np.loadtxt('mode_one_data_hydro.txt')
mode_others_array_hydro = np.loadtxt('mode_others_data_hydro.txt')

mode_zero_array_first_order = np.loadtxt('mode_zero_data_first_order.txt')
mode_one_array_first_order = np.loadtxt('mode_one_data_first_order.txt')
mode_others_array_first_order = np.loadtxt('mode_others_data_first_order.txt')

p2_start = -np.pi
p2_end   =  np.pi

tau_array = np.append(tau_mr, np.flip(tau_mc))
tau_array = np.append(tau_array, tau_mr_first_order)

pl.figure(figsize=[12,7.5])

count = 0
for tau in tau_array:
    print ('Count : ', count)

    if (count < plot_breaks[0]):
        tau = tau_mr[count]
        tau_ee = tau
        tau_eph = 100.0
        filename = 'ohmic_ballistic/f_vs_theta_1_0_tau_mr_%.2f.txt'%tau
        if count < change_N_at:
           N_p2 = 1024
        else:
           N_p2 = 8192
    
    elif(count > plot_breaks[0]) and (count < plot_breaks[1]):
        tau = tau_mc[tau_mc.size-(count-plot_breaks[0])]
        tau_ee = np.inf
        tau_eph = tau
        filename = 'hydro_ballistic/f_vs_theta_1_0_tau_mr_%.2f.txt'%tau
        if (tau_mc.size - (count-plot_breaks[0])) < change_N_at:
            N_p2 = 1024
        else:
            N_p2 = 8192

    elif (count > plot_breaks[1]):
        tau = tau_mr_first_order[tau_mr_first_order.size - (count-plot_breaks[1])]
        tau_ee = 0.01
        tau_eph = tau
        filename = 'ohmic_hydro/f_vs_theta_1_0_tau_mr_%.2f.txt'%tau

    f = np.loadtxt(filename)
    p2 = p2_start + (0.5 + np.arange(N_p2)) * (p2_end - p2_start)/N_p2

    radius = f.copy()/np.max(np.abs(f))
    theta  = p2.copy()

    bg_radius = 5.

    x = (radius + bg_radius)* np.cos(theta)
    y = (radius + bg_radius)* np.sin(theta)

    x_bg = bg_radius * np.cos(theta)
    y_bg = bg_radius * np.sin(theta)

    gs = gridspec.GridSpec(4, 4)
    ax = pl.subplot(gs[:3, :3])

    # Put this under if conditions
    if count < plot_breaks[0]:
        ax.set_title('$\mathrm{\\tau_{mc} = \infty\ (ps),\ \\tau_{mr} = %.2f\ (ps)}$'%tau)
    elif (count > plot_breaks[0]) and (count < plot_breaks[1]):
        ax.set_title('$\mathrm{\\tau_{mc} = %.2f\ (ps),\ \\tau_{mr} = 100.00\ (ps)}$'%tau)
    elif(count > plot_breaks[1]):
        ax.set_title('$\mathrm{\\tau_{mc} = 0.01\ (ps),\ \\tau_{mr} = %.2f\ (ps)}$'%tau)

    ax.plot(x, y, color='C3')
    ax.plot(x_bg, y_bg, color='black', alpha=0.2)

    #pl.xticks([])
    #pl.yticks([])

    ax.set_aspect('equal')

    ax.set_xlim([-(1.3*bg_radius), + 1.3*bg_radius])
    ax.set_ylim([-(1.3*bg_radius), + 1.3*bg_radius])

    ax2 = pl.subplot(gs[3, 0])

    ax2.loglog((tau_mr), mode_zero_array_ohmic, '-', label = '$A$', lw=3)
    ax2.loglog((tau_mr), mode_one_array_ohmic, '-', label = '$B$', lw=3)
    ax2.loglog((tau_mr), mode_others_array_ohmic, '-', label='Fluctuations', lw=3)

    ax2.set_xlim([0.01, 100.])
    ax2.set_ylim([1e-7, 1e-1])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines['right'].set_visible(False)


    ax3 = pl.subplot(gs[3, 1])
    ax3.loglog((tau_mc), mode_zero_array_hydro, label = '$A$', lw=3)
    ax3.loglog((tau_mc), mode_one_array_hydro, label = '$B$', lw=3)
    ax3.loglog((tau_mc), mode_others_array_hydro, label='Fluctuations', lw=3)

    ax3.axvline(100, color = 'k', alpha = 0.5, linestyle = '--' , linewidth=1)

    ax3.set_xlim([0.01, 100.])
    ax3.set_ylim([1e-7, 1e-1])
    ax3.invert_xaxis()
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.spines['left'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    ax4 = pl.subplot(gs[3, 2])

    ax4.loglog((tau_mr_first_order), mode_zero_array_first_order, '-', label = '$A$', lw=3)
    ax4.loglog((tau_mr_first_order), mode_one_array_first_order, '-', label = '$B$', lw=3)
    ax4.loglog((tau_mr_first_order), mode_others_array_first_order, '-', label='Fluctuations', lw=3)
    ax4.axvline(100, color = 'k', alpha = 0.5, linestyle = '--', linewidth=1 )

    ax4.set_xlim([0.01, 100.])
    ax4.set_ylim([1e-7, 1e-1])
    ax4.invert_xaxis()
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.spines['left'].set_visible(False)

    if (count < plot_breaks[0]):
        ax2.axvline(tau_mr[count], color='k')
    elif(count > plot_breaks[0]) and (count < plot_breaks[1]):
        ax3.axvline(tau_mc[tau_mc.size-(count-plot_breaks[0])], color='k')
    elif (count > plot_breaks[1]):
        ax4.axvline(tau_mr_first_order[tau_mr_first_order.size - (count-plot_breaks[1])], color='k')

    #pl.tight_layout()
    pl.subplots_adjust(wspace=0.01)
    pl.subplots_adjust(hspace=0.5)
    pl.savefig('images/dump_%06d.png'%count)
    pl.clf()
    count = count + 1






