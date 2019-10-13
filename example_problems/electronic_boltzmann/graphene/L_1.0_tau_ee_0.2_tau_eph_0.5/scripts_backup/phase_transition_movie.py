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
pl.rcParams['font.size']       = 20

pl.rcParams['font.sans-serif'] = 'serif'
pl.rcParams['text.usetex']     = True
pl.rcParams['axes.linewidth']  = 1.5
pl.rcParams['axes.titlesize']  = 16
pl.rcParams['axes.labelsize']  = 20

pl.rcParams['xtick.major.size'] = 8
pl.rcParams['xtick.minor.size'] = 4
pl.rcParams['xtick.major.pad']  = 8
pl.rcParams['xtick.minor.pad']  = 8
pl.rcParams['xtick.color']      = 'k'
pl.rcParams['xtick.labelsize']  = 20
pl.rcParams['xtick.direction']  = 'in'

pl.rcParams['ytick.major.size'] = 8
pl.rcParams['ytick.minor.size'] = 4
pl.rcParams['ytick.major.pad']  = 8
pl.rcParams['ytick.minor.pad']  = 8
pl.rcParams['ytick.color']      = 'k'
pl.rcParams['ytick.labelsize']  = 20
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
print ("change at : ", change_N_at)

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
print ("breaks : ", plot_breaks)

mode_zero_array_ohmic = np.loadtxt('mode_zero_data.txt')
mode_one_array_ohmic = np.loadtxt('mode_one_data.txt')
mode_others_array_ohmic = np.loadtxt('mode_others_data.txt')

mode_zero_array_hydro = np.loadtxt('mode_zero_data_hydro.txt')
mode_one_array_hydro = np.loadtxt('mode_one_data_hydro.txt')
mode_others_array_hydro = np.loadtxt('mode_others_data_hydro.txt')

mode_zero_array_first_order = np.loadtxt('mode_zero_data_first_order.txt')
mode_one_array_first_order = np.loadtxt('mode_one_data_first_order.txt')
mode_others_array_first_order = np.loadtxt('mode_others_data_first_order.txt')


# Clean up a glitchy datapoint in the first order transition
delete_index = -50
print ("delete index : ", delete_index)
tau_mr_first_order = np.delete(tau_mr_first_order, delete_index)

mode_zero_array_first_order = np.delete(mode_zero_array_first_order, delete_index)
mode_one_array_first_order = np.delete(mode_one_array_first_order, delete_index)
mode_others_array_first_order = np.delete(mode_others_array_first_order, delete_index)


p2_start = -np.pi
p2_end   =  np.pi

tau_array = np.append(tau_mr, np.flip(tau_mc))
tau_array = np.append(tau_array, tau_mr_first_order)


count = 701
for tau in tau_array[count:count+1]:
    print ('Count : ', count, ", tau : ", tau)

    if (count < plot_breaks[0]):
        tau = tau_mr[count]
        print ('Inside tau1 : ', tau)
        tau_ee = tau
        tau_eph = 100.0
        filename = 'ohmic_ballistic_f_vs_tau/f_vs_theta_0_6_tau_mr_%.2f.txt'%tau
        filepath = \
            '/home/mchandra/gitansh/gitansh_bolt/example_problems/electronic_boltzmann/L_1.0_2.5_tau_ee_inf_tau_eph_%.2f_DC/dumps'%tau

        if count < change_N_at:
           N_p2 = 1024
        else:
           N_p2 = 8192
    
    elif(count >= plot_breaks[0]) and (count < plot_breaks[1]):
        #tau = tau_mc[tau_mc.size-(count-plot_breaks[0])]
        tau = np.flip(tau_mc)[count-plot_breaks[0]]
        print ('Inside tau2 : ', tau)
        tau_ee = np.inf
        tau_eph = tau
        filename = 'hydro_ballistic_f_vs_tau/f_vs_theta_0_6_tau_mr_%.2f.txt'%tau
        filepath = \
            '/home/mchandra/gitansh/gitansh_bolt/example_problems/electronic_boltzmann/L_1.0_2.5_tau_ee_%.2f_tau_eph_100.00_DC/dumps'%tau
        
        if (tau_mc.size - (count-plot_breaks[0])) <= change_N_at:
            N_p2 = 1024
        else:
            N_p2 = 8192

    elif (count >= plot_breaks[1]):
        N_p2 = 1024
        #tau = tau_mr_first_order[tau_mr_first_order.size - (count-plot_breaks[1])]
        tau = np.flip(tau_mr_first_order)[count-plot_breaks[1]]
        print ('Inside tau3 : ', tau)
        tau_ee = 0.01
        tau_eph = tau
        filename = 'ohmic_hydro_f_vs_tau/f_vs_theta_0_6_tau_mr_%.2f.txt'%tau
        filepath = \
            '/home/mchandra/gitansh/gitansh_bolt/example_problems/electronic_boltzmann/L_1.0_2.5_tau_ee_0.01_tau_eph_%.2f_DC/dumps'%tau

    
    moment_files = np.sort(glob.glob(filepath+'/moment*.h5'))
    lagrange_multiplier_files = np.sort(glob.glob(filepath+'/lagrange_multipliers*.h5'))

    h5f     = h5py.File(moment_files[-1], 'r')
    moments = np.swapaxes(h5f['moments'][:], 0, 1)
    h5f.close()

    density = moments[:, :, 0]
    j_x     = moments[:, :, 1]
    j_y     = moments[:, :, 2]

    h5f  = h5py.File(lagrange_multiplier_files[-1], 'r')
    lagrange_multipliers = h5f['lagrange_multipliers'][:]
    h5f.close()

    mu    = lagrange_multipliers[:, :, 0]
    mu_ee = lagrange_multipliers[:, :, 1]
    T_ee  = lagrange_multipliers[:, :, 2]
    vel_drift_x = lagrange_multipliers[:, :, 3]
    vel_drift_y = lagrange_multipliers[:, :, 4]

    f = np.loadtxt(filename)
    p2 = p2_start + (0.5 + np.arange(N_p2)) * (p2_end - p2_start)/N_p2

    radius = f.copy()/np.max(np.abs(f))
    theta  = p2.copy()

    bg_radius = 5.

    x = (radius + bg_radius)* np.cos(theta)
    y = (radius + bg_radius)* np.sin(theta)

    x_bg = bg_radius * np.cos(theta)
    y_bg = bg_radius * np.sin(theta)
    
    # Put this under if conditions
    if count < plot_breaks[0]:
        pl.suptitle('$\mathrm{\\tau_{mc} = \infty\ (ps),\ \\tau_{mr} = %.2f\ (ps)}$'%tau,\
                y = 0.05)
    elif (count >= plot_breaks[0]) and (count < plot_breaks[1]):
        pl.suptitle('$\mathrm{\\tau_{mc} = %.2f\ (ps),\ \\tau_{mr} = 100.00\ (ps)}$'%tau,\
                y = 0.05)
    elif(count >= plot_breaks[1]):
        pl.suptitle('$\mathrm{\\tau_{mc} = 0.01\ (ps),\ \\tau_{mr} = %.2f\ (ps)}$'%tau,\
                y = 0.05)


    gs = gridspec.GridSpec(15, 15)
    ax = pl.subplot(gs[:9, 1:9])

    #ax.plot(x, y, color='C3')
    ax.plot(x_bg, y_bg, color='black', alpha=0.2)

    pl.gca().set_xticklabels([])
    pl.gca().set_yticklabels([])

    ax.set_aspect('equal')

    ax.set_xlim([-(1.3*bg_radius), + 1.3*bg_radius])
    ax.set_ylim([-(1.3*bg_radius), + 1.3*bg_radius])

    ax2 = pl.subplot(gs[11:, :5])

    ax2.loglog((tau_mr), mode_zero_array_ohmic, '-', label = '$A$', lw=3)
    ax2.loglog((tau_mr), mode_one_array_ohmic, '-', label = '$B$', lw=3)
    ax2.loglog((tau_mr), mode_others_array_ohmic, '-', label='Fluctuations', lw=3)

    ax2.set_xlim([0.01, 100.])
    ax2.set_ylim([1e-7, 1e-1])
    ax2.set_xticks([0.01, 1])
    #ax2.set_yticks([])
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.spines['right'].set_visible(False)
    #ax2.set_xlabel('$\\tau_{\mathrm{mr}}$')
    #ax2.set_ylabel('$\mathrm{Mode\ Amplitude}$', fontsize=16)


    ax3 = pl.subplot(gs[11:, 5:10])
    ax3.loglog((tau_mc), mode_zero_array_hydro, label = '$A$', lw=3)
    ax3.loglog((tau_mc), mode_one_array_hydro, label = '$B$', lw=3)
    ax3.loglog((tau_mc), mode_others_array_hydro, label='Fluctuations', lw=3)

    ax3.axvline(100, color = 'k', alpha = 0.5, linestyle = '--' , linewidth=1)

    ax3.set_xlim([0.01, 100.])
    ax3.set_ylim([1e-7, 1e-1])
    ax3.invert_xaxis()
    ax3.set_xticks([0.01, 1, 100])
    ax3.set_yticks([])
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    ax3.spines['left'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    #ax3.set_xlabel('$\\tau_{\mathrm{mc}}$')
    

    ax4 = pl.subplot(gs[11:, 10:15])

    ax4.loglog((tau_mr_first_order), mode_zero_array_first_order, '-', label = '$A$', lw=3)
    ax4.loglog((tau_mr_first_order), mode_one_array_first_order, '-', label = '$B$', lw=3)
    ax4.loglog((tau_mr_first_order), mode_others_array_first_order, '-', label='Fluctuations', lw=3)
    ax4.axvline(100, color = 'k', alpha = 0.5, linestyle = '--', linewidth=1 )
    #ax4.axvline(tau_mr_first_order[delete_index], color='k', lw=1 )

    ax4.set_xlim([0.01, 100.])
    ax4.set_ylim([1e-7, 1e-1])
    ax4.invert_xaxis()
    ax4.set_xticks([0.01, 1])
    ax4.set_yticks([])
    ax4.set_xticklabels([])
    ax4.set_yticklabels([])
    ax4.spines['left'].set_visible(False)
    #ax4.set_xlabel('$\\tau_{\mathrm{mr}}$')
    #ax4.legend(bbox_to_anchor=(1.0, 1),prop={'size':13})

    #if (count < plot_breaks[0]):
    #    ax2.axvline(tau_mr[count], color='k')
    #elif(count > plot_breaks[0]) and (count < plot_breaks[1]):
    #    ax3.axvline(tau_mc[tau_mc.size-(count-plot_breaks[0])], color='k')
    #elif (count > plot_breaks[1]):
    #    ax4.axvline(tau_mr_first_order[tau_mr_first_order.size - (count-plot_breaks[1])], color='k')
    if (count < plot_breaks[0]):
        ax2.axvline(tau, color='k')
    elif(count >= plot_breaks[0]) and (count < plot_breaks[1]):
        ax3.axvline(tau, color='k')
    elif (count >= plot_breaks[1]):
        ax4.axvline(tau, color='k')

    ax5 = pl.subplot(gs[0:9, 9:])

    ax5.contourf(q1_meshgrid, q2_meshgrid, density, 100, cmap='bwr')
    ax5.streamplot(q1, q2, 
                   vel_drift_x, vel_drift_y,
                   density=2, color='k',
                   linewidth=0.7, arrowsize=1
                  )
    ax5.set_xlim([q1[0], q1[-1]])
    ax5.set_ylim([1.25, q2[-1]])
    #ax5.set_yticks([1.5, 2.2])
    #ax5.set_xticks([0.2, 0.8])
    ax5.set_yticks([])
    ax5.set_xticks([])
    #ax5.set_xlabel("$x\ (\mathrm{\mu m})$", labelpad=-15)
    #ax5.set_ylabel("$y\ (\mathrm{\mu m})$", labelpad=-15)
    ax5.set_aspect('equal')
    #ax5.yaxis.tick_right()
    #ax5.yaxis.set_label_position("right")


    #pl.tight_layout()
    pl.subplots_adjust(wspace=0.02)
    pl.subplots_adjust(hspace=-0.2)
    #pl.savefig('images/dump_%06d.png'%count, bbox_inches = 'tight',\
    #        ad_inches=0.1)
    pl.savefig('images/dump_%06d.png'%count)
    pl.clf()
    count = count + 1






