import sys
import arrayfire as af
import numpy as np
import pylab as pl
import h5py
import petsc4py, sys; petsc4py.init(sys.argv)
from petsc4py import PETSc

from bolt.lib.physical_system import physical_system

from bolt.lib.nonlinear_solver.nonlinear_solver \
    import nonlinear_solver
from bolt.lib.nonlinear_solver.EM_fields_solver.electrostatic \
    import compute_electrostatic_fields

# Horizontal
import domain_1
import boundary_conditions_1
import params_1
import initialize_1

# Vertical
import domain_2
import boundary_conditions_2
import params_2
import initialize_2

import bolt.src.electronic_boltzmann.advection_terms as advection_terms

import bolt.src.electronic_boltzmann.collision_operator \
    as collision_operator

import bolt.src.electronic_boltzmann.moment_defs as moment_defs



# Defining the physical system to be solved:
system_1 = physical_system(domain_1,
                         boundary_conditions_1,
                         params_1,
                         initialize_1,
                         advection_terms,
                         collision_operator.RTA,
                         moment_defs
                        )

system_2 = physical_system(domain_2,
                         boundary_conditions_2,
                         params_2,
                         initialize_2,
                         advection_terms,
                         collision_operator.RTA,
                         moment_defs
                        )

# Time parameters:
# TODO : For physically coupled simulations, dt_1 == dt_2, and t_final_1 == t_final_2
# Only one index for t0 and time_step required.
dt_1                  = params_1.dt
t_final_1             = params_1.t_final
params_1.current_time = t0_1        = 0.0
params_1.time_step    = time_step_1 = 0

dt_2                  = params_2.dt
t_final_2             = params_2.t_final
params_2.current_time = t0_2        = 0.0
params_2.time_step    = time_step_2 = 0


dump_counter = 0
dump_time_array = []

N_g_1  = domain_1.N_ghost
N_g_2  = domain_2.N_ghost

# Declaring a nonlinear system object which will evolve the defined physical system:
nls_1 = nonlinear_solver(system_1)
params_1.rank = nls_1._comm.rank

nls_2 = nonlinear_solver(system_2)
params_2.rank = nls_2._comm.rank

if (params_1.restart):
    nls_1.load_distribution_function(params_1_.restart_file)
if (params_2.restart):
    nls_2.load_distribution_function(params_2_.restart_file)

density_1 = nls_1.compute_moments('density')
print("rank = ", params_1.rank, "\n",
      "     <mu>    = ", af.mean(params_1.mu[0, N_g_1:-N_g_1, N_g_1:-N_g_1]), "\n",
      "     max(mu) = ", af.max(params_1.mu[0, N_g_1:-N_g_1, N_g_1:-N_g_1]), "\n",
      "     <n>     = ", af.mean(density_1[0, N_g_1:-N_g_1, N_g_1:-N_g_1]), "\n",
      "     max(n)  = ", af.max(density_1[0, N_g_1:-N_g_1, N_g_1:-N_g_1]), "\n"
     )

density_2 = nls_2.compute_moments('density')
print("rank = ", params_2.rank, "\n",
      "     <mu>    = ", af.mean(params_2.mu[0, N_g_2:-N_g_2, N_g_2:-N_g_2]), "\n",
      "     max(mu) = ", af.max(params_2.mu[0, N_g_2:-N_g_2, N_g_2:-N_g_2]), "\n",
      "     <n>     = ", af.mean(density_2[0, N_g_2:-N_g_2, N_g_2:-N_g_2]), "\n",
      "     max(n)  = ", af.max(density_2[0, N_g_2:-N_g_2, N_g_2:-N_g_2]), "\n"
     )

#nls.f     = af.select(nls.f < 1e-20, 1e-20, nls.f)

while t0_1 < t_final_1:

    # Refine to machine error
    if (time_step_1==0):
        params_1.collision_nonlinear_iters = 10
    else:
        params_1.collision_nonlinear_iters = params_1.collision_operator_nonlinear_iters
    
    if (time_step_2==0):
        params_2.collision_nonlinear_iters = 10
    else:
        params_2.collision_nonlinear_iters = params_2.collision_operator_nonlinear_iters

    dump_steps_1 = params_1.dump_steps
    dump_steps_2 = params_2.dump_steps
    # Uncomment if need to dump more frequently during a desired time interval
    #if (params.current_time > 149. and params.current_time < 154):
    #    dump_steps = 1
    #else:
    #    dump_steps = params.dump_steps
    if (time_step_1%params_1.dump_dist_after==0):
        file_number = '%06d'%dump_counter
        nls_1.dump_distribution_function('dumps/f_1_' + file_number)
        nls_2.dump_distribution_function('dumps/f_2_' + file_number)

    if (time_step_1%dump_steps_1==0):
        file_number = '%06d'%dump_counter
        dump_counter= dump_counter + 1
        dump_time_array.append(params_1.current_time)
        PETSc.Sys.Print("=====================================================")
        PETSc.Sys.Print("Dumping data at time step =", time_step_1,
                         ", file number =", file_number
                       )
        PETSc.Sys.Print("=====================================================")
        if (params_1.rank==0):
            np.savetxt("dump_time_array.txt", dump_time_array)

        nls_1.dump_moments('dumps/moments_1_' + file_number)
        nls_2.dump_moments('dumps/moments_2_' + file_number)

        if(time_step_1 == 0):
            nls_1.dump_distribution_function('dumps/f_1_' + file_number)
            nls_2.dump_distribution_function('dumps/f_2_' + file_number)

        nls_1.dump_aux_arrays([params_1.mu,
                             params_1.mu_ee,
                             params_1.T_ee,
                             params_1.vel_drift_x, params_1.vel_drift_y,
                             params_1.j_x, params_1.j_y],
                             'lagrange_multipliers',
                             'dumps/lagrange_multipliers_1_' + file_number
                            )
        nls_2.dump_aux_arrays([params_2.mu,
                             params_2.mu_ee,
                             params_2.T_ee,
                             params_2.vel_drift_x, params_2.vel_drift_y,
                             params_2.j_x, params_2.j_y],
                             'lagrange_multipliers',
                             'dumps/lagrange_multipliers_2_' + file_number
                            )

    dt_force_constraint = 0.
#    dt_force_constraint = \
#        0.5 * np.min(nls.dp1, nls.dp2) \
#            / np.max((af.max(nls.cell_centered_EM_fields[0]),
#                      af.max(nls.cell_centered_EM_fields[1])
#                     )
#                    )


    PETSc.Sys.Print("Time step =", time_step_1, ", Time =", t0_1)

    nls_1.strang_timestep(dt_1)
    nls_2.strang_timestep(dt_2)
    t0_1                = t0_1 + dt_1
    t0_2                = t0_2 + dt_2
    time_step_1         = time_step_1 + 1
    time_step_2         = time_step_2 + 1
    params_1.time_step    = time_step_1
    params_2.time_step    = time_step_2
    params_1.current_time = t0_1
    params_2.current_time = t0_2

    # Floors
    #nls.f     = af.select(nls.f < 1e-20, 1e-20, nls.f)

    density_1 = nls_1.compute_moments('density')
    density_2 = nls_2.compute_moments('density')
    print("rank = ", params_1.rank, "\n",
          "     <mu>    = ", af.mean(params_1.mu[0, N_g_1:-N_g_1, N_g_1:-N_g_1]), "\n",
          "     max(mu) = ", af.max(params_1.mu[0, N_g_1:-N_g_1, N_g_1:-N_g_1]), "\n",
          "     <n>     = ", af.mean(density_1[0, N_g_1:-N_g_1, N_g_1:-N_g_1]), "\n",
          "     max(n)  = ", af.max(density_1[0, N_g_1:-N_g_1, N_g_1:-N_g_1]), "\n"
         )
    PETSc.Sys.Print("--------------------\n")
    print("rank = ", params_2.rank, "\n",
          "     <mu>    = ", af.mean(params_2.mu[0, N_g_2:-N_g_2, N_g_2:-N_g_2]), "\n",
          "     max(mu) = ", af.max(params_2.mu[0, N_g_2:-N_g_2, N_g_2:-N_g_2]), "\n",
          "     <n>     = ", af.mean(density_2[0, N_g_2:-N_g_2, N_g_2:-N_g_2]), "\n",
          "     max(n)  = ", af.max(density_2[0, N_g_2:-N_g_2, N_g_2:-N_g_2]), "\n"
         )
    PETSc.Sys.Print("--------------------\n")

nls_1.dump_distribution_function('dumps/f_1_laststep')
nls_2.dump_distribution_function('dumps/f_2_laststep')
