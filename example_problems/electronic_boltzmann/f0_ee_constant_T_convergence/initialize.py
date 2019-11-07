"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np
from petsc4py import PETSc

def initialize_f(q1, q2, p1, p2, p3, params):

    PETSc.Sys.Print("Initializing f")
    k = params.boltzmann_constant

    params.mu          = 0.*q1 + params.initial_mu
    params.T           = 0.*q1 + params.initial_temperature

    params.vel_drift_x = 0.*q1
    params.vel_drift_y = 0.*q1
    params.phi         = 0.*q1

    params.mu_ee       = params.mu.copy()
    params.T_ee        = params.T.copy()

    # Initialize with non-zero velocities
    params.vel_drift_x = 0.*q1 + 1e-2
    params.vel_drift_y = 0.*q1 + 5e-2

    params.E_band   = params.band_energy(p1, p2)
    params.vel_band = params.band_velocity(p1, p2)

    E_upper = params.E_band + params.charge_electron*params.phi

    p_x = params.initial_mu * p1**0 * af.cos(p2)
    p_y = params.initial_mu * p1**0 * af.sin(p2)

    f = (1./(af.exp( (E_upper - params.vel_drift_x*p_x
                              - params.vel_drift_y*p_y
                              - params.mu
                    )/(k*params.T)
                  ) + 1.
           ))

    af.eval(f)
    return(f)


def initialize_E(q1, q2, params):

    E1 = 0.*q1
    E2 = 0.*q1
    E3 = 0.*q1

    return(E1, E2, E3)

def initialize_B(q1, q2, params):

    B1 = 0.*q1
    B2 = 0.*q1
    B3 = 0.*q1

    return(B1, B2, B3)