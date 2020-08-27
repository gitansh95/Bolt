import numpy as np
import pylab as pl

import domain
import params

fermi_velocity      = params.fermi_velocity
boltzmann_constant  = params.boltzmann_constant
mu                  = params.initial_mu
initial_temperature = params.initial_temperature

vel_drift_x         = 0.
vel_drift_y         = 0.

p_F                 = mu/fermi_velocity # For linear dispersion

N_p1 = domain.N_p1
N_p2 = domain.N_p2

p1_start = domain.p1_start
p1_end   = domain.p1_end
p2_start = domain.p2_start
p2_end   = domain.p2_end

p1           = p1_start[0] + (0.5 + np.arange(N_p1)) * (p1_end[0] - p1_start[0])/N_p1
p1_numerical = p1_start[0] + (0.5 + np.arange(N_p1)) * (p1_end[0] - p1_start[0])/N_p1
p2           = p2_start[0] + (0.5 + np.arange(N_p2)) * (p2_end[0] - p2_start[0])/N_p2

p1, p2 = np.meshgrid(p1, p2)

p_x = p1 * np.cos(p2)
p_y = p1 * np.sin(p2)

p   = np.sqrt(p_x**2. + p_y**2.)

# For linear dispersion
E_band_linear = p*fermi_velocity

f_linear = (1./(np.exp( (E_band_linear - vel_drift_x*p_x
                                       - vel_drift_y*p_y
                                       - mu
                        )/(boltzmann_constant*initial_temperature)
                      ) + 1.
                ))

g = (E_band_linear - vel_drift_x*p1*np.cos(p2) - vel_drift_y*p1*np.sin(p2) - mu)/(initial_temperature*boltzmann_constant)

dg_dp1 = (1 - vel_drift_x*np.cos(p2) - vel_drift_y*np.sin(p2))/(initial_temperature*boltzmann_constant)
df_dp1 = -(np.exp(g)/((np.exp(g)+1)**2)) * dg_dp1

df_dp1_numerical = np.loadtxt("d_flux_p1_dp1.txt")
print (df_dp1_numerical.shape)


pl.plot(p1_numerical, df_dp1_numerical, '-o', label = "Linear")
pl.plot(p1[0, :], df_dp1[0, :], '-o', label = "Linear")
pl.plot(p1[0, :], np.gradient(f_linear[0, :], p1[0, :]) , '-o', label = "Linear")

pl.xlabel("p$_r$")
pl.ylabel("f")

pl.savefig("iv.png")
