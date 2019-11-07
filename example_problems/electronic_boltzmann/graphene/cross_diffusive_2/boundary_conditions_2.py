import numpy as np
import arrayfire as af
import params as params_common
import domain as domain_common

from bolt.src.electronic_boltzmann.collision_operator import f0_defect_constant_T

in_q1_left   = 'mirror+dirichlet'
in_q1_right  = 'mirror+dirichlet'
in_q2_bottom = 'mirror+dirichlet'
in_q2_top    = 'mirror+dirichlet'

# Interface to the device on the left + diffusive on the rest
#@af.broadcast
def f_left(f, q1, q2, p1, p2, p3, params):

    N_g = domain_common.N_ghost

    fermi_dirac_1 = params_common.f_1
    print ("fermi_dirac_1 : ", fermi_dirac_1.shape)
    print ('f _ 2 : ', f.shape)
    
    contact_width = fermi_dirac_1.shape[2] - 2*N_g
    q2_index = (f.shape[2]/2) - contact_width/2

    f_left = f
    f_left[:, :N_g, q2_index:q2_index+contact_width] = fermi_dirac_1[:, -2*N_g:-N_g, N_g:-N_g]
    
    
    N_g = domain_common.N_ghost
    f0  = f0_defect_constant_T(f, p1, p2, p3, params)

    f_left[:, :N_g, :q2_index]               = f0[:, :N_g, :q2_index]
    f_left[:, :N_g, q2_index+contact_width:] = f0[:, :N_g, q2_index+contact_width:]


    af.eval(f_left)
    return(f_left)

# Interface to the device on the right + diffusive on the rest
#@af.broadcast
def f_right(f, q1, q2, p1, p2, p3, params):

    N_g = domain_common.N_ghost

    fermi_dirac_3 = params_common.f_3
    print ("fermi_dirac_3 : ", fermi_dirac_3.shape)
    print ('f _ 2 : ', f.shape)
    
    contact_width = fermi_dirac_3.shape[2] - 2*N_g
    q2_index = f.shape[2]/2 - contact_width/2

    f_right = f
    f_right[:, -N_g:, q2_index:q2_index+contact_width] = fermi_dirac_3[:, N_g:2*N_g, N_g:-N_g]
    
    N_g = domain_common.N_ghost
    f0  = f0_defect_constant_T(f, p1, p2, p3, params)

    f_right[:, -N_g:, :q2_index]               = f0[:, -N_g:, :q2_index]
    f_right[:, -N_g:, q2_index+contact_width:] = f0[:, -N_g:, q2_index+contact_width:]

    af.eval(f_right)
    return(f_right)

# Outward contact
@af.broadcast
def f_bottom(f, q1, q2, p1, p2, p3, params):

    k       = params.boltzmann_constant
    E_upper = params.E_band
    T       = params.initial_temperature
    mu      = params.initial_mu
    
    t     = params.current_time
    omega = 2. * np.pi * params.AC_freq
    vel_drift_x_in  = -params.vel_drift_x_in
    
    p_x = mu * p1**0 * af.cos(p2)
    p_y = mu * p2**0 * af.sin(p2)

    fermi_dirac_in = (1./(af.exp( (E_upper - vel_drift_x_in*p_y - mu)/(k*T) ) + 1.)
                     )

    q1_contact_start = 0.0 #params.contact_start
    q1_contact_end   = 0.5#params.contact_end
        
    cond = ((q1 >= q1_contact_start) & \
            (q1 <= q1_contact_end) \
           )

    f_bottom = cond*fermi_dirac_in + (1 - cond)*f

               
    af.eval(f_bottom)
    return(f_bottom)

# Diffusuve bcs
#@af.broadcast
def f_top(f, q1, q2, p1, p2, p3, params):
    
    N_g = domain_common.N_ghost

    f0 = f0_defect_constant_T(f, p1, p2, p3, params)

    f_top = f
    f_top[:, :, -N_g:] = f0[:, :, -N_g:]
    
    af.eval(f_top)
    return(f_top)
