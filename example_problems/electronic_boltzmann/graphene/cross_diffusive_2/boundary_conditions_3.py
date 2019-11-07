import numpy as np
import arrayfire as af
import params as params_common
import domain as domain_common

from bolt.src.electronic_boltzmann.collision_operator import f0_defect_constant_T

in_q1_left   = 'mirror+dirichlet'
in_q1_right  = 'mirror+dirichlet'
in_q2_bottom = 'mirror+dirichlet'
in_q2_top    = 'mirror+dirichlet'

# Interface to device on the left
@af.broadcast
def f_left(f, q1, q2, p1, p2, p3, params):

    N_g = domain_common.N_ghost

    fermi_dirac_2 = params_common.f_2
    print ("fermi_dirac_2 : ", fermi_dirac_2.shape)
    print ('f _ 3 : ', f.shape)
        
    contact_width = f.shape[2] - 2*N_g
    q2_index = fermi_dirac_2.shape[2]/2 - contact_width/2

    f_left = f
    f_left[:, :N_g, N_g:-N_g] = \
            fermi_dirac_2[:, -2*N_g:-N_g, q2_index:q2_index+contact_width] 

    af.eval(f_left)
    return(f_left)

# Diffusive bcs
#@af.broadcast
def f_right(f, q1, q2, p1, p2, p3, params):
    
    N_g = domain_common.N_ghost

    f0 = f0_defect_constant_T(f, p1, p2, p3, params)

    f_right = f
    f_right[:, -N_g:, :] = f0[:, -N_g:, :]
    
    af.eval(f_right)
    return(f_right)

# Diffusive bcs
#@af.broadcast
def f_top(f, q1, q2, p1, p2, p3, params):
    
    N_g = domain_common.N_ghost

    f0 = f0_defect_constant_T(f, p1, p2, p3, params)

    f_top = f
    f_top[:, :, -N_g:] = f0[:, :, -N_g:]
    
    af.eval(f_top)
    return(f_top)

# Diffusive bcs
#@af.broadcast
def f_bottom(f, q1, q2, p1, p2, p3, params):
    
    N_g = domain_common.N_ghost

    f0 = f0_defect_constant_T(f, p1, p2, p3, params)

    f_bottom = f
    f_bottom[:, :, :N_g] = f0[:, :, :N_g]
    
    af.eval(f_bottom)
    return(f_bottom)


