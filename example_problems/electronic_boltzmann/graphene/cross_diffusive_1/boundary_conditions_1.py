import numpy as np
import arrayfire as af
import params as params_common
import domain as domain_common

from bolt.src.electronic_boltzmann.collision_operator import f0_defect_constant_T

in_q1_left   = 'mirror+dirichlet'
in_q1_right  = 'mirror+dirichlet'
in_q2_bottom = 'mirror+dirichlet'
in_q2_top    = 'mirror+dirichlet'

# Inward current
@af.broadcast
def f_left(f, q1, q2, p1, p2, p3, params):

    k       = params.boltzmann_constant
    E_upper = params.E_band
    T       = params.initial_temperature
    mu      = params.initial_mu
    
    t     = params.current_time
    omega = 2. * np.pi * params.AC_freq
    vel_drift_x_in  = params.vel_drift_x_in
    
    p_x = mu * p1**0 * af.cos(p2)
    p_y = mu * p2**0 * af.sin(p2)

    fermi_dirac_in = (1./(af.exp( (E_upper - vel_drift_x_in*p_x - mu)/(k*T) ) + 1.)
                     )

    if (params.contact_geometry=="straight"):
        # Contacts on either side of the device

        q2_contact_start = 0.0 #params.contact_start
        q2_contact_end   = 0.5#params.contact_end
        
        cond = ((q2 >= q2_contact_start) & \
                (q2 <= q2_contact_end) \
               )

        f_left = cond*fermi_dirac_in + (1 - cond)*f

    elif (params.contact_geometry=="turn_around"):
        # Contacts on the same side of the device

        vel_drift_x_out = -params.vel_drift_x_in * np.sin(omega*t)

        fermi_dirac_out = (1./(af.exp( (E_upper - vel_drift_x_out*p_x - mu)/(k*T) ) + 1.)
                          )
    
        # TODO: set these parameters in params.py
        cond_in  = ((q2 >= 3.5) & (q2 <= 4.5))
        cond_out = ((q2 >= 5.5) & (q2 <= 6.5))
    
        f_left =  cond_in*fermi_dirac_in + cond_out*fermi_dirac_out \
                + (1 - cond_in)*(1 - cond_out)*f
                

    af.eval(f_left)
    return(f_left)

# Interface to the device on the right
@af.broadcast
def f_right(f, q1, q2, p1, p2, p3, params):

    N_g = domain_common.N_ghost

    fermi_dirac_2 = params_common.f_2
    contact_width = f.shape[2] - 2*N_g
    q2_index = int((fermi_dirac_2.shape[2])/2 - contact_width/2)

    
    f_right = f
    
    f_right[:, -N_g:, N_g:-N_g] = \
            fermi_dirac_2[:, N_g:2*N_g, q2_index:q2_index+contact_width]

    af.eval(f_right)
    return(f_right)


# Diffusuve bcs
@af.broadcast
def f_top(f, q1, q2, p1, p2, p3, params):
    
    k       = params.boltzmann_constant
    E_upper = params.E_band
    T       = params.initial_temperature
    mu      = params.initial_mu
    
    t     = params.current_time
    omega = 2. * np.pi * params.AC_freq
    vel_drift_x_in  = 0.*params.vel_drift_x_in
    
    p_x = mu * p1**0 * af.cos(p2)
    p_y = mu * p2**0 * af.sin(p2)

    fermi_dirac_in = (1./(af.exp( (E_upper - vel_drift_x_in*p_x - mu)/(k*T) ) + 1.)
                     )

    q1_contact_start = 0.0 #params.contact_start
    q1_contact_end   = 0.75#params.contact_end
    
    cond = ((q1 >= q1_contact_start) & \
            (q1 <= q1_contact_end) \
           )

    f_top = cond*fermi_dirac_in + (1 - cond)*f
    
    af.eval(f_top)
    return(f_top)

# Diffusive bcs
@af.broadcast
def f_bottom(f, q1, q2, p1, p2, p3, params):
    
    k       = params.boltzmann_constant
    E_upper = params.E_band
    T       = params.initial_temperature
    mu      = params.initial_mu
    
    t     = params.current_time
    omega = 2. * np.pi * params.AC_freq
    vel_drift_x_in  = 0.*params.vel_drift_x_in
    
    p_x = mu * p1**0 * af.cos(p2)
    p_y = mu * p2**0 * af.sin(p2)

    fermi_dirac_in = (1./(af.exp( (E_upper - vel_drift_x_in*p_x - mu)/(k*T) ) + 1.)
                     )

    q1_contact_start = 0.0 #params.contact_start
    q1_contact_end   = 0.75#params.contact_end
    
    cond = ((q1 >= q1_contact_start) & \
            (q1 <= q1_contact_end) \
           )

    f_bottom = cond*fermi_dirac_in + (1 - cond)*f
    
    af.eval(f_bottom)
    return(f_bottom)
