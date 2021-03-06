import arrayfire as af
import numpy as np

from bolt.lib.physical_system import physical_system
from bolt.lib.linear.linear_solver import linear_solver

import physics_tests.nonrelativistic_boltzmann.domain as domain
import physics_tests.nonrelativistic_boltzmann.boundary_conditions as boundary_conditions
import physics_tests.nonrelativistic_boltzmann.params as params
import physics_tests.nonrelativistic_boltzmann.initialize as initialize

import bolt.src.nonrelativistic_boltzmann.advection_terms as advection_terms
import bolt.src.nonrelativistic_boltzmann.collision_operator as collision_operator
import bolt.src.nonrelativistic_boltzmann.moments as moments

N = 2**np.arange(5, 10)

def run_cases(q_dim, p_dim, charge_electron, tau):

    params.charge[0] = charge_electron
    params.tau       = tau

    # Running the setup for all resolutions:
    for i in range(N.size):
        af.device_gc()
        domain.N_q1 = int(N[i])

        if(q_dim == 2):
            domain.N_q2 = int(N[i])
            params.k_q2 = 4 * np.pi

        if(p_dim >= 2):
         
            domain.N_p2     = 32
            domain.p2_start = -10
            domain.p2_end   = 10

        if(p_dim == 3):

            domain.N_p3     = 32
            domain.p3_start = -10
            domain.p3_end   = 10


        if(charge_electron != 0):
            domain.N_p1 = int(N[i])

            if(p_dim >= 2):
                domain.N_p2 = int(N[i])

            if(p_dim == 3):
                domain.N_p3 = int(N[i])

        params.p_dim = p_dim

        # Defining the physical system to be solved:
        system = physical_system(domain,
                                 boundary_conditions,
                                 params,
                                 initialize,
                                 advection_terms,
                                 collision_operator.BGK,
                                 moments
                                )

        ls = linear_solver(system)

        # Timestep as set by the CFL condition:
        dt = params.N_cfl * min(ls.dq1, ls.dq2) \
                          / max(domain.p1_end, domain.p2_end, domain.p3_end)

        time_array = np.arange(dt, params.t_final + dt, dt)
        # Checking that time array doesn't cross final time:
        if(time_array[-1]>params.t_final):
            time_array = np.delete(time_array, -1)

        for time_index, t0 in enumerate(time_array):
            ls.RK4_timestep(dt)

        ls.dump_distribution_function('dump_files/lsf_' + str(N[i]))
