import numpy as np
import pylab as pl
from scipy.optimize import curve_fit
import scipy.fftpack

pl.rcParams['figure.figsize']  = 12, 7.5
pl.rcParams['lines.linewidth'] = 1.5
pl.rcParams['font.family']     = 'serif'
pl.rcParams['font.weight']     = 'bold'
pl.rcParams['font.size']       = 20

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

p2_start = -np.pi
p2_end   =  np.pi
N_p2     = 1024 #use 8192 for tau_mr >= 0.20, and 1024 for tau_mr < 0.20

N_samples = N_p2
step_size = (p2_end - p2_start)/N_p2

p2 = p2_start + (0.5 + np.arange(N_p2)) * (p2_end - p2_start)/N_p2
theta  = p2.copy()

N = 20
angle_step = 2*np.pi/N
counter = 0

angle_1 = 0.

fig = pl.figure()

while angle_1 < 2*np.pi:
    angle_2 = 0.
    while angle_2 < 2*np.pi:
        for angle_3 in np.arange(0.0, 2*np.pi, angle_step):
            # Test for fft
            test_array = 5 + np.cos(theta + angle_1) + np.cos(2*theta + angle_2) + np.cos(3*theta + angle_3)

            x_test = test_array * np.cos(theta)
            y_test = test_array * np.sin(theta)

            bg_radius = 5
            x_bg = bg_radius * np.cos(theta)
            y_bg = bg_radius * np.sin(theta)

            x_mode_1 = (bg_radius + np.cos(theta + angle_1))*np.cos(theta)
            y_mode_1 = (bg_radius + np.cos(theta + angle_1))*np.sin(theta)

            x_mode_2 = (bg_radius + np.cos(2*theta + angle_2))*np.cos(theta)
            y_mode_2 = (bg_radius + np.cos(2*theta + angle_2))*np.sin(theta)

            x_mode_3 = (bg_radius + np.cos(3*theta + angle_3))*np.cos(theta)
            y_mode_3 = (bg_radius + np.cos(3*theta + angle_3))*np.sin(theta)
            
            ax = fig.add_subplot(2, 3, 1)
            pl.plot(x_mode_1, y_mode_1, color='red')
            pl.plot(x_bg, y_bg, color='black', alpha=0.2)
            pl.plot(0., 0., 'o', markersize=5, color='black')
            pl.xlim([-8, 8])
            pl.ylim([-8, 8])
            #pl.xticks([])
            #pl.yticks([])
            
            ax.set_aspect('equal')
            ax.set_title('$\mathrm{n = 1,\ \\theta = %.2f}$'%angle_1)

            ax2 = fig.add_subplot(2, 3, 2)
            pl.plot(x_mode_2, y_mode_2, color='red')
            pl.plot(x_bg, y_bg, color='black', alpha=0.2)
            pl.plot(0., 0., 'o', markersize=5, color='black')
            ax2.set_aspect('equal')
            ax2.set_title('$\mathrm{n = 2,\ \\theta = %.2f}$'%angle_2)
            pl.xlim([-8, 8])
            pl.ylim([-8, 8])
            pl.xticks([])
            pl.yticks([])

            ax3 = fig.add_subplot(2,3, 3)
            pl.plot(x_mode_3, y_mode_3, color='red')
            pl.plot(x_bg, y_bg, color='black', alpha=0.2)
            pl.plot(0., 0., 'o', markersize=5, color='black')
            ax3.set_aspect('equal')
            ax3.set_title('$\mathrm{n = 3,\ \\theta = %.2f}$'%angle_3)
            pl.xlim([-8, 8])
            pl.ylim([-8, 8])
            #pl.xticks([])
            pl.yticks([])


            ax4 = fig.add_subplot(2, 3, 5)
            pl.plot(x_test, y_test, color='red')
            pl.plot(x_bg, y_bg, color='black', alpha=0.2)
            pl.plot(0., 0., 'o', markersize=5, color='black')
            ax4.set_aspect('equal')
            ax4.set_title('$\mathrm{\delta f}$')
            pl.xlim([-8, 8])
            pl.ylim([-8, 8])

            #pl.tight_layout()
            pl.subplots_adjust(wspace=0.01, hspace=0.19)
            pl.savefig('images/mode_movie_%06d.png'%counter)
            pl.clf()
            counter = counter + 1
            angle_2 = angle_2 + angle_step/10
            angle_1 = angle_1 + angle_step/20

