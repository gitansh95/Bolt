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
pl.rcParams['figure.figsize']  = 13*1.5, 7.5*1.5
pl.rcParams['figure.dpi']      = 300
pl.rcParams['image.cmap']      = 'jet'
pl.rcParams['lines.linewidth'] = 1.5
pl.rcParams['font.family']     = 'serif'
pl.rcParams['font.weight']     = 'normal'
pl.rcParams['font.size']       = 24
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

pl.rcParams['lines.antialiased'] = True

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


# Define the dimensions of the 2x5 array being plotted
N_q1_temp = 72*2
q1_start_temp = 0.0
q1_end_temp = 2.0
q1_temp = q1_start_temp + \
        (0.5 + np.arange(N_q1_temp)) * (q1_end_temp - q1_start_temp)/N_q1_temp
    
q2_start_temp  = 0.0
q2_end_temp = 5.0
N_q2_temp = int(round(5.0*N_q1))
q2_temp   = q2_start_temp + \
        (0.5 + np.arange(N_q2_temp)) * (q2_end_temp - q2_start_temp)/N_q2_temp


time_step = 0.025/4
dump_step = 5

####

heights_1 = np.arange(0.25, 5.01, 0.05)
heights_2 = np.arange(5.25, 10.001, 0.25)

heights   = heights_1
heights = np.append(heights, heights_2)

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
        np.load('edge_data_tau_mr_inf/edge_L_1.0_%.2f_tau_ee_inf_tau_eph_inf.txt.npz'%index)
    voltage_left = voltages['left']
    voltage_right = voltages['right']

    currents = \
        np.loadtxt('edge_data_tau_mr_inf/current_L_1.0_%.2f_tau_ee_inf_tau_eph_inf.txt'%index)
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


for number, index in enumerate(heights_2[:]):
    print ('Index : ', index)

    filepath = \
    '/home/mchandra/cci_data/dumps_ballistic_tau_inf_L_1.0_%.1f'%index
    moment_files 		  = np.sort(glob.glob(filepath+'/moment*.h5'))
    lagrange_multiplier_files = \
            np.sort(glob.glob(filepath+'/lagrange_multipliers*.h5'))

    file_number = -2
    print (filepath)
    
    q2_start = 0.0
    q2_end = index/2
    N_q2 = int(round(q2_end*N_q1))
    q2 = q2_start + (0.5 + np.arange(N_q2)) * (q2_end - q2_start)/N_q2

    moments_file = moment_files[file_number]
    moments = io.readBinaryFile(moments_file)
    moments = moments[0].reshape(N_q2, N_q1, 3)
    density = moments[:, :, 0]

    lagrange_multipliers_file = lagrange_multiplier_files[file_number]
    lagrange_multipliers = io.readBinaryFile(lagrange_multipliers_file)
    lagrange_multipliers = lagrange_multipliers[0].reshape(N_q2, N_q1, 7)
    j_x_2 = lagrange_multipliers[:, :, 5]
    j_y_2 = lagrange_multipliers[:, :, 6]

    
    delta_n = density - np.mean(density)

    # Plot and layout
    N_cols = 1#int(round(4/q2_end))
    gs = gridspec.GridSpec(6, N_cols)

   
    ax1 = pl.subplot(gs[:4, 0])
    
    density_full_array = np.zeros((int(N_q2_full/2), 2*N_q1))
    j_x_2_full_array   = np.zeros((int(N_q2_full/2), 2*N_q1))
    j_y_2_full_array   = np.zeros((int(N_q2_full/2), 2*N_q1))

    # If plot of data fits in the top row
    if (N_q2 <= int(N_q2_full/2)):
        density_full_array[:N_q2, :N_q1] = delta_n
        j_x_2_full_array[:N_q2, :N_q1]   = j_x_2
        j_y_2_full_array[:N_q2, :N_q1]   = j_y_2
    
        ax1.axvline(q2_start,
                ymin=q1_temp[int(N_q1_temp/2)-1]/q1_end_temp,
                ymax=q1_temp[-1]/q1_end_temp,
                color='k', lw=2) #Left spine
    
        ax1.axvline(q2_end,
                ymin=q1_temp[int(N_q1_temp/2)-1]/q1_end_temp,
                ymax=q1_temp[-1]/q1_end_temp,
                color='k', lw=2) #Right spine
    
        ax1.axhline(-q1_temp[0],
                xmin=q2_start/q2_end_temp,
                xmax=q2_end/q2_end_temp,
                color='k', lw=2) # Top spine

        ax1.axhline(-q1_temp[int(N_q1_temp/2)],
                xmin=q2_start/q2_end_temp,
                xmax=q2_end/q2_end_temp,
                color='k', lw=2) #Bottom spine

    # If plot of data requires 2 rows
    else:
        density_full_array[:, :N_q1] = delta_n[:int(N_q2_full/2), :]
        j_x_2_full_array[:, :N_q1]   = j_x_2[:int(N_q2_full/2), :]
        j_y_2_full_array[:, :N_q1]   = j_y_2[:int(N_q2_full/2), :]
        
        density_full_array[:N_q2-int(N_q2_full/2), N_q1:] = \
                delta_n[int(N_q2_full/2):, :]
        j_x_2_full_array[:N_q2-int(N_q2_full/2), N_q1:]   = \
                j_x_2[int(N_q2_full/2):, :]
        j_y_2_full_array[:N_q2-int(N_q2_full/2), N_q1:]   = \
                j_y_2[int(N_q2_full/2):, :]
    
        ax1.axvline(q2_start,
                ymin=q1_temp[int(N_q1_temp/2)-1]/q1_end_temp,
                ymax=q1_temp[-1]/q1_end_temp,
                color='k', lw=2) #Left spine
    
        ax1.axvline(q2_end,
                ymin=q1_temp[int(N_q1_temp/2)-1]/q1_end_temp,
                ymax=q1_temp[-1]/q1_end_temp,
                color='k', lw=2) #Right spine
        
        ax1.axvline(q2_end-q2_end_temp,
                ymax=q1_temp[int(N_q1_temp/2)-1]/q1_end_temp,
                ymin=q1_temp[0]/q1_end_temp,
                color='k', lw=2) #Right spine
    
        ax1.axhline(-q1_temp[0],
                color='k', lw=2) # Top spine

        ax1.axhline(-q1_temp[int(N_q1_temp/2)],
                color='k', lw=2) #Mid spine
        
        ax1.axhline(-q1_temp[-1],
                xmin=q2_start/q2_end_temp,
                xmax=(q2_end-q2_end_temp)/q2_end_temp,
                color='k', lw=2) #Bottom spine

    
    
    base = ax1.transData
    rot = transforms.Affine2D().rotate_deg(-90)

    density_min = -11000
    density_max = 11000

    im = ax1.contourf(q1_temp, q2_temp, density_full_array, 200,
                norm=colors.SymLogNorm(linthresh=density_full_array.max()/20,
                    vmin=density_min, vmax=density_max), cmap='bwr',
                    transform = rot + base)
    
    #im = ax1.contourf(q1_temp, q2_temp,
    #            density_full_array, 200,
    #            norm = MidpointNormalize(midpoint=0, vmin=density_min, vmax=density_max),
    #            cmap='bwr', transform = rot + base
    #                 )

    ax1.streamplot(q1_temp, q2_temp,
                j_x_2_full_array,
                j_y_2_full_array,
                density=8, color='black',
                linewidth=1.2, arrowsize=1.4, transform = rot + base
                  )

    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['right'].set_visible(False)


    #ax1.set_xlim([0, q2_end])
    ax1.set_xlim([0, 5.0])
    ax1.set_ylim([-2, 0])

    #ax1.set_xlabel(r'$y\;(\mu \mathrm{m})$', labelpad = 0)
    #ax1.set_ylabel(r'$x\;(\mu \mathrm{m})$')

    #ax3 = ax1.twiny()  # instantiate a second axes that shares the same y-axis
    #ax3.tick_params(axis='x', direction = 'in', pad = 0.04)
    #labels = [item.get_text() for item in ax3.get_xticklabels()]
    #if (q2_end > 1.0):
    #    labels[1]= 1.0
    #if(q2_end > 2.0):
    #    labels[2]= 2.0
    #if(q2_end > 3.0):
    #    labels[3]= 3.0
    #if(q2_end > 4.0):
    #    labels[4]= 4.0
    #ax3.set_xticklabels(labels)

    #ax1.tick_params(direction = 'in', pad = 3.5)
    #labels = [item.get_text() for item in ax1.get_xticklabels()]
    #if(q2_end > 6.0):
    #    labels[1]= 6.0
    #if(q2_end > 7.0):
    #    labels[2]= 7.0
    #if(q2_end > 8.0):
    #    labels[3]= 8.0
    #if(q2_end > 9.0):
    #    labels[4]= 9.0
    #ax1.set_xticklabels(labels)
    
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    ax1.set_aspect('equal')
    
    #ax1.spines['top'].set_linewidth(0)
    #ax1.spines['bottom'].set_linewidth(0)
    #ax1.spines['left'].set_linewidth(0)
    #ax1.spines['right'].set_linewidth(0)


    ax2 = pl.subplot(gs[4:, 0])
    
    norm = np.max(resistance_local_list_tau_mr_inf_vel_1em3)
    ax2.plot(heights - 0.25, \
        (resistance_local_list_tau_mr_inf_vel_1em3/norm), \
        marker = 'o', markersize=4, lw=2, label="$\mathrm{\\tau_{mr} = \infty,\ vel = 1e^{-3} }$")
    
    ax2.axhline(1, color='C0', linestyle='--', lw=3)
    ax2.plot(heights[number]-0.25,
            resistance_local_list_tau_mr_inf_vel_1em3[number]/norm,
            color='red', marker='+', mew = 5,  markersize=15)

    ax2.set_xlim(xmin=0.0, xmax=9.75)
    ax2.set_ylim(ymax=1.10)

    #ax2.set_xlabel(r'Aspect Ratio ($\Gamma$)')
    #ax2.set_ylabel(r'Resistance (a. u.)')
    ax2.text(8.85, 0.05, "$ %.2f$"%(heights[number]))
    ax2.xaxis.set_ticklabels([])
    ax2.yaxis.set_ticklabels([])


    #pl.suptitle('Quantum Critical Ballistic Transport in Two-Dimensional Fermi Liquids\nChandra, Kataria, & Sahdev (2019)',
    #             fontsize = 24)

    pl.subplots_adjust(bottom=0.1, right=0.85, top=0.9, hspace=0.5)
    cax = pl.axes([0.87, 0.1, 0.01, 0.8])
    cbar = pl.colorbar(im, cax=cax)
    cbar.set_ticks([])
    #pl.gcf().text(0.88, 0.08, "-V")
    #pl.gcf().text(0.88, 0.90, "+V")

    #pl.tight_layout()

    #pl.savefig('images/dump_tau_mr_inf_laststep_L_1.0_%.2f.png'%q2_end)
    pl.savefig('images/dump_%06d.png'%number)
    pl.clf()
    
