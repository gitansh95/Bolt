import numpy as np
import arrayfire as af
import domain

from bolt.src.electronic_boltzmann.collision_operator import f0_defect_constant_T

in_q1_left   = 'mirror+dirichlet'
in_q1_right  = 'mirror+dirichlet'
in_q2_bottom = 'mirror+dirichlet'
in_q2_top    = 'mirror+dirichlet'

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
        
    q2_0 = (params.contact_end - params.contact_start)/2
    spatial_profile = -(q2 - q2_0)**2 + q2_0**2
    spatial_profile = spatial_profile/np.max(np.abs(spatial_profile))

    fermi_dirac_in = (1./(af.exp( (E_upper - vel_drift_x_in*p_x*spatial_profile - mu)/(k*T) ) + 1.)
                     )

    if (params.contact_geometry=="straight"):
        # Contacts on either side of the device

        q2_contact_start = params.contact_start
        q2_contact_end   = params.contact_end
        
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

@af.broadcast
def f_right(f, q1, q2, p1, p2, p3, params):

    k       = params.boltzmann_constant
    E_upper = params.E_band
    T       = params.initial_temperature
    mu      = params.initial_mu

    t     = params.current_time
    omega = 2. * np.pi * params.AC_freq
    vel_drift_x_out = params.vel_drift_x_out
    
    p_x = mu * p1**0 * af.cos(p2)
    p_y = mu * p2**0 * af.sin(p2)
        
    q2_0 = (params.contact_end - params.contact_start)/2
    spatial_profile = -(q2 - q2_0)**2 + q2_0**2
    spatial_profile = spatial_profile/np.max(np.abs(spatial_profile))
    print ('boundary_conditions.py, spatial_profile : ', spatial_profile.shape)
    print ('boundary_conditions.py, p_x : ', p_x.shape)

    fermi_dirac_out = (1./(af.exp( (E_upper - vel_drift_x_out*p_x*spatial_profile - mu)/(k*T) ) + 1.)
                      )
    
    if (params.contact_geometry=="straight"):
        # Contacts on either side of the device

        q2_contact_start = params.contact_start
        q2_contact_end   = params.contact_end
        
        cond = ((q2 >= q2_contact_start) & \
                (q2 <= q2_contact_end) \
               )

        f_right = cond*fermi_dirac_out + (1 - cond)*f

    elif (params.contact_geometry=="turn_around"):
        # Contacts on the same side of the device
        
        f_right = f

    af.eval(f_right)
    return(f_right)

#@af.broadcast
def f_top(f, q1, q2, p1, p2, p3, params):
    
    N_g = domain.N_ghost

    f0 = f0_defect_constant_T(f, p1, p2, p3, params)

    f_top = f
    f_top[:, :, -N_g:] = f0[:, :, -N_g:]
    
    af.eval(f_top)
    return(f_top)

#@af.broadcast
def f_bottom(f, q1, q2, p1, p2, p3, params):
    
    N_g = domain.N_ghost

    f0 = f0_defect_constant_T(f, p1, p2, p3, params)

    f_bottom = f
    f_bottom[:, :, :N_g] = f0[:, :, :N_g]
    
    af.eval(f_bottom)
    return(f_bottom)

#@af.broadcast
#def f_top(f, q1, q2, p1, p2, p3, params):

#    k       = params.boltzmann_constant
#    E_upper = params.E_band
#    T       = params.initial_temperature
#    mu      = params.initial_mu
    
#    t     = params.current_time
#    omega = 2. * np.pi * params.AC_freq
#    vel_drift_x_in  = params.vel_drift_x_in*0.
    
#    p_x = mu * p1**0 * af.cos(p2)
#    p_y = mu * p2**0 * af.sin(p2)
        
#    fermi_dirac_in = (1./(af.exp( (E_upper - vel_drift_x_in*p_x - mu)/(k*T) ) + 1.)
#                     )

#    q1_contact_start = 0.
#    q1_contact_end   = 10.0
        
#    cond = ((q1 >= q1_contact_start) & \
#            (q1 <= q1_contact_end) \
#           )

#    f_top = cond*fermi_dirac_in + (1 - cond)*f

#    af.eval(f_top)
#    return(f_top)

#@af.broadcast
#def f_bottom(f, q1, q2, p1, p2, p3, params):

#    k       = params.boltzmann_constant
#    E_upper = params.E_band
#    T       = params.initial_temperature
#    mu      = params.initial_mu
#    
#    t     = params.current_time
#    omega = 2. * np.pi * params.AC_freq
#    vel_drift_x_in  = params.vel_drift_x_in*0.
#    
#    p_x = mu * p1**0 * af.cos(p2)
#    p_y = mu * p2**0 * af.sin(p2)
#        
#    fermi_dirac_in = (1./(af.exp( (E_upper - vel_drift_x_in*p_x - mu)/(k*T) ) + 1.)
#                     )
#
#    q1_contact_start = 0.
#    q1_contact_end   = 10.0
#        
#    cond = ((q1 >= q1_contact_start) & \
#            (q1 <= q1_contact_end) \
#           )
#
#    f_bottom = cond*fermi_dirac_in + (1 - cond)*f
#
#    af.eval(f_bottom)
#    return(f_bottom)
