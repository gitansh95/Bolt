import numpy as np
import scipy.fftpack
import csv
import pylab as pl
from scipy.optimize import curve_fit
from scipy.optimize import root
from scipy.signal import correlate
from scipy.interpolate import interp1d

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

def line (x, m, c):
    return (m*x + c)

def sin_curve_fit(t, A, tau_in):
        return A*np.sin(2*np.pi*AC_freq*(t + tau_in ))

lengths = np.arange(0.75, 0.751, 0.25)
freq = np.arange(2.0, 100.0, 1.0)
AC_freq_array = freq/100.0

phase_arrays = []
zero_crossing_array = []
for l in lengths:
    L1 = l
    L2 = 2.5*l

    phase_vs_freq_array_1 = []
    phase_vs_freq_array_2 = []
    for index in freq:

        AC_freq = index/100.0
        time_period = 1/AC_freq

        time         = np.loadtxt("frequency_scaling/time_L_%.3f_%.3f_tau_ee_inf_tau_eph_5.0_freq_%.1f.txt"%(L1, L2, index))
        edge_density = np.loadtxt("frequency_scaling/edge_density_L_%.3f_%.3f_tau_ee_inf_tau_eph_5.0_freq_%.1f.txt"%(L1, L2, index))
        q2           = np.loadtxt("frequency_scaling/q2_edge_L_%.3f_%.3f_tau_ee_inf_tau_eph_5.0_freq_%.1f.txt"%(L1, L2, index))

        #time         = np.loadtxt("frequency_scaling/time_L_%.3f_%.3f_tau_ee_inf_tau_eph_0.1_freq_%.1f.txt"%(L1, L2, index))
        #edge_density = np.loadtxt("frequency_scaling/edge_density_L_%.3f_%.3f_tau_ee_inf_tau_eph_0.1_freq_%.1f.txt"%(L1, L2, index))
        #q2           = np.loadtxt("frequency_scaling/q2_edge_L_%.3f_%.3f_tau_ee_inf_tau_eph_0.1_freq_%.1f.txt"%(L1, L2, index))

        #time         = np.loadtxt("frequency_scaling/time_L_%.3f_%.3f_tau_ee_0.1_tau_eph_10.0_freq_%.1f.txt"%(L1, L2, index))
        #edge_density = np.loadtxt("frequency_scaling/edge_density_L_%.3f_%.3f_tau_ee_0.1_tau_eph_10.0_freq_%.1f.txt"%(L1, L2, index))
        #q2           = np.loadtxt("frequency_scaling/q2_edge_L_%.3f_%.3f_tau_ee_0.1_tau_eph_10.0_freq_%.1f.txt"%(L1, L2, index))

        print (edge_density.shape)

        half_time = time.shape[0]/2
        N_spatial = edge_density.shape[1]\

        drive = np.sin(2*np.pi*AC_freq*time)
        nsamples = drive.size
        dt_corr = np.linspace(-time[-1] + time[0],\
                time[-1] - time[0], 2*nsamples-1)

        # Discarding transients
        q = q2.size/2
        time_half = time[half_time:]
        drive_half = drive[half_time:]

        phase_shift_corr_array = []
        phase_shift_fitting_array = []\

        for i in range(N_spatial):
            #print ('index : ', i)
            signal_1 = edge_density[:, i]
            norm_1 = np.max(signal_1[half_time:])
            signal_1_normalized = signal_1/norm_1

            # Calculate phase_shifts using scipy.correlate
            corr = correlate(drive, signal_1_normalized)
            time_shift_corr = dt_corr[corr.argmax()]
            phase_shift_corr  = 2*np.pi*(((0.5 + time_shift_corr/time_period) % 1.0) - 0.5)

            # Calculate phase_shifts using scipy.curve_fit
            popt, pcov = curve_fit(sin_curve_fit, time[half_time:],\
                        signal_1_normalized[half_time:])
            time_shift_fitting = popt[1]%(time_period/2.0)
            phase_shift_fitting  = 2*np.pi*(((0.5 + time_shift_fitting/time_period) % 1.0) - 0.5)

            phase_shift_corr_array.append(phase_shift_corr)
            phase_shift_fitting_array.append(phase_shift_fitting)

        phase_shift_corr_array = np.array(phase_shift_corr_array)
        phase_shift_fitting_array = np.array(phase_shift_fitting_array)

        tolerance = 1.6
        tolerance_2 = np.pi#3.14

        while (np.any(phase_shift_fitting_array - phase_shift_corr_array > tolerance)):
            print (np.max(np.abs(phase_shift_fitting_array - phase_shift_corr_array)))
            for i in range(len(q2)):
                if (phase_shift_fitting_array[i] - phase_shift_corr_array[i] > tolerance):
                    phase_shift_fitting_array[i] = phase_shift_fitting_array[i] - np.pi
                if (phase_shift_corr_array[i] - phase_shift_fitting_array[i] > tolerance):
                    phase_shift_fitting_array[i] = phase_shift_fitting_array[i] + np.pi

        for i in range(len(q2)-1):
            while (np.abs(phase_shift_corr_array[i] - phase_shift_corr_array[i+1]) > tolerance_2):
                print ((phase_shift_corr_array[i] - phase_shift_corr_array[i+1]))
                if (phase_shift_corr_array[i] - phase_shift_corr_array[i+1] > tolerance_2):
                    phase_shift_corr_array[i+1] = phase_shift_corr_array[i+1] + 2*np.pi
                    phase_shift_fitting_array[i+1] = phase_shift_fitting_array[i+1] + 2*np.pi
                if (phase_shift_corr_array[i+1] - phase_shift_corr_array[i] > tolerance_2):
                    phase_shift_corr_array[i+1] = phase_shift_corr_array[i+1] - 2*np.pi
                    phase_shift_fitting_array[i+1] = phase_shift_fitting_array[i+1] - 2*np.pi

        phase_vs_freq_array_1.append(phase_shift_fitting_array[0])
        phase_vs_freq_array_2.append(phase_shift_fitting_array[-1])

        pl.plot(q2, phase_shift_corr_array, '--o', label = "$\mathrm{Corr}$", color = 'cyan', alpha = 0.5)
        pl.plot(q2, phase_shift_fitting_array, '-', label = "$\mathrm{Fit}$", color = 'b', alpha = 0.5)

        pl.axhline(0, color = 'k', ls = '--')

        # Plot
        pl.ylabel('$\mathrm{\phi}$')
        pl.xlabel('$\mathrm{y\ \mu m}$')

        #pl.title('$1.0 \\times 2.5,\ \\tau_\mathrm{mr} = 5.0,\ \\tau_ \mathrm{mc} = \infty}$')
        pl.legend(loc='best')
        pl.tight_layout()
        pl.savefig('images/phase_vs_y_L_%.3f_%.3f_tau_ee_inf_tau_eph_5.0_freq_%.1f.png'%(L1, L2, index))
        #pl.savefig('images/phase_vs_y_L_%.3f_%.3f_tau_ee_0.1_tau_eph_10.0_freq_%1f.png'%(L1, L2, index))
        pl.clf()

    phase_vs_freq_array_1 = np.array(phase_vs_freq_array_1)
    phase_vs_freq_array_2 = np.array(phase_vs_freq_array_2)
    phase_arrays.append(phase_vs_freq_array_2)

    #zero_crossing_array.append(np.where(np.diff(np.sign(phase_vs_freq_array_2)))[0][0])

#zero_crossing_array = np.array(zero_crossing_array)
## Get the tau_mrs corresponding to the zero crossing
#fit_index_1 = np.array(AC_freq_array[zero_crossing_array])
#fit_index_2 = np.array(AC_freq_array[(zero_crossing_array+1)])

phase_arrays = np.array(phase_arrays)

## Do not constrain phase to remain within -pi and pi
for l in range(lengths.size):
    for f in range(freq.size-1):
        if ((phase_arrays[l, f] - phase_arrays[l, f+1]) > tolerance_2):
            phase_arrays[l, f+1] = phase_arrays[l, f+1] + 2*np.pi
        if ((phase_arrays[l, f+1] - phase_arrays[l, f]) > tolerance_2):
            phase_arrays[l, f+1] = phase_arrays[l, f+1] - 2*np.pi


#np.savetxt('images/zero_crossing_array.txt', zero_crossing_array)
#np.savetxt('fit_index_1.txt', fit_index_1)
#np.savetxt('fit_index_2.txt', fit_index_2)
#np.savetxt('images/freq.txt', AC_freq_array)

#pl.subplot(211)
#for i in range(lengths.size):
    #pl.plot(AC_freq_array, phase_arrays[i, :], '-o', label = "L = %.3f"%(lengths[i]), alpha = 0.5)
    #np.savetxt('images/phase_vs_freq_L_%.3f_%.3f.txt'%(lengths[i], 2.5*lengths[i]), phase_arrays[i, :])
    ##pl.plot(AC_freq_array, phase_vs_freq_array_2, '-o', label = "$\mathrm{Top\ Edge}$", color = 'cyan', alpha = 0.5)

#pl.axhline(0, color = 'k', ls = '--')

## Plot
#pl.ylabel('$\mathrm{\phi}$')
#pl.xlabel('$\mathrm{f\ (ps^{-1})}$')

#pl.legend(loc='best', prop={'size':10})

#pl.subplot(212)

## Fit lines to the 2 points surrounding the zero crossing
#f_array = []
#for i in range(lengths.size):
    #j = zero_crossing_array[i]
    ##print (fit_index_1[i], fit_index_2[i])
    ##print (V_vs_tau_array[i][j], V_vs_tau_array[i][j+1])
    #x = [fit_index_1[i], fit_index_2[i]]
    #print (x)
    #y = [phase_arrays[i, j], phase_arrays[i, j+1]]
    #f = interp1d(x, y, fill_value="extrapolate")
    ##pl.plot(x, y, color = 'k', alpha = 0.5, ls='--')
    #f_array.append(f)

#f_array = np.array(f_array)
#root_array = []
#count = 0
#starting_guess = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
#for item in f_array:
    #print (count)
    #guess = starting_guess[count]
    #count = count+1
    #root_array.append(root(item, guess).x[0])

#root_array = np.array(root_array)

#pl.plot(lengths, 1/root_array, '-o')

##popt, pcov = curve_fit(line, lengths, root_array)
##pl.text(0.45, 0.35, 'Slope = %.2f'%popt[0])
##pl.plot(lengths, line(lengths, popt[0], popt[1]), color = 'k', ls= '--')

#pl.ylabel('$1/f_\mathrm{\phi_{0}}\ (ps^{-1})$')
#pl.xlabel('$L_{x}\ (\mu m)$')

#pl.tight_layout()
#pl.savefig('images/phase_vs_freq.png')
#pl.clf()




for i in range(lengths.size):

    flag = True
    while (flag):
        flag = False
        for index in range(1, AC_freq_array.size):
            #print ("Index ", index)
            if ((phase_arrays[i, index] - phase_arrays[i, index-1]) > tolerance):
                #print ("Freq ", AC_freq_array[index])
                phase_arrays[i, index] = phase_arrays[i, index] - np.pi
                flag = True

    pl.plot(AC_freq_array, phase_arrays[i, :], '-o', label = "L = %.3f"%(lengths[i]), alpha = 0.5)
    np.savetxt('images/phase_vs_freq_L_%.3f_%.3f.txt'%(lengths[i], 2.5*lengths[i]), phase_arrays[i, :])
    #pl.plot(AC_freq_array, phase_vs_freq_array_2, '-o', label = "$\mathrm{Top\ Edge}$", color = 'cyan', alpha = 0.5)

pl.axhline(0, color = 'k', ls = '--')

# Plot
pl.ylabel('$\mathrm{\phi}$')
pl.xlabel('$\mathrm{f\ (ps^{-1})}$')

pl.legend(loc='best', prop={'size':10})
pl.savefig('images/phase_vs_freq.png')
pl.clf()
