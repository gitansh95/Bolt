import sys
import arrayfire as af
import numpy as np
import pylab as pl
import h5py
#import petsc4py, sys; petsc4py.init(sys.argv)
#from petsc4py import PETSc

from bolt.lib.physical_system import physical_system

from bolt.lib.nonlinear_solver.nonlinear_solver \
    import nonlinear_solver
from bolt.lib.nonlinear_solver.EM_fields_solver.electrostatic \
    import compute_electrostatic_fields
from bolt.lib.nonlinear_solver.compute_moments \
   import compute_moments as compute_moments_imported

import domain
import boundary_conditions
import params
import initialize

import bolt.src.electronic_boltzmann.advection_terms as advection_terms

import bolt.src.electronic_boltzmann.collision_operator \
    as collision_operator

import bolt.src.electronic_boltzmann.moment_defs as moment_defs


# Defining the physical system to be solved:
system = physical_system(domain,
                         boundary_conditions,
                         params,
                         initialize,
                         advection_terms,
                         collision_operator.RTA,
                         moment_defs
                        )

# Time parameters:
dt      = params.dt
t_final = params.t_final
params.current_time = t0        = 0.0
params.time_step    = time_step = 0
dump_counter = 0

# Declaring a nonlinear system object which will evolve the defined physical system:
nls = nonlinear_solver(system)
params.rank = nls._comm.rank
N_g = domain.N_ghost

density = nls.compute_moments('density')

h5f = h5py.File('dumps/000000.h5', 'w')
h5f.create_dataset('n', data = density[:, N_g:-N_g, N_g:-N_g])
h5f.close()

time_array = np.arange(dt, t_final+dt, dt)

print("rank = ", params.rank, "\n",
          "     <mu>    = ", af.mean(params.mu[0, N_g:-N_g, N_g:-N_g]), "\n",
          "     max(mu) = ", af.max(params.mu[0, N_g:-N_g, N_g:-N_g]), "\n",
          "     <mu_ee>    = ", af.mean(params.mu_ee[0, N_g:-N_g, N_g:-N_g]), "\n",
          "     max(mu_ee) = ", af.max(params.mu_ee[0, N_g:-N_g, N_g:-N_g]), "\n",
          "     T_ee    = ", af.mean(params.T_ee[0, N_g:-N_g, N_g:-N_g]), "\n",
          #"     <phi>   = ", af.mean(params.phi[N_g:-N_g, N_g:-N_g]), "\n",
          "     <n>     = ", af.mean(density[0, N_g:-N_g, N_g:-N_g]), "\n",
          "     max(n)  = ", af.max(density[0, N_g:-N_g, N_g:-N_g]), "\n",
          "     |E1|    = ", af.mean(af.abs(nls.cell_centered_EM_fields[0, N_g:-N_g, N_g:-N_g])),
          "\n",
          "     |E2|    = ", af.mean(af.abs(nls.cell_centered_EM_fields[1, N_g:-N_g, N_g:-N_g]))
         )
print("--------------------\n")

for time_index, t0 in enumerate(time_array):

    # Refine to machine error
    if (time_index==0):
        params.collision_nonlinear_iters = 10
    else:
        params.collision_nonlinear_iters = params.collision_operator_nonlinear_iters


    collision_operator.RTA(nls.f, nls.q1_center, nls.q2_center, \
        nls.p1, nls.p2, nls.p3, compute_moments_imported, params)
    print ('Stepped through collision operator')

    density = nls.compute_moments('density')
    print("rank = ", params.rank, "\n",
          "     <mu>    = ", af.mean(params.mu[0, N_g:-N_g, N_g:-N_g]), "\n",
          "     max(mu) = ", af.max(params.mu[0, N_g:-N_g, N_g:-N_g]), "\n",
          "     <mu_ee>    = ", af.mean(params.mu_ee[0, N_g:-N_g, N_g:-N_g]), "\n",
          "     max(mu_ee) = ", af.max(params.mu_ee[0, N_g:-N_g, N_g:-N_g]), "\n",
          "     T_ee    = ", af.mean(params.T_ee[0, N_g:-N_g, N_g:-N_g]), "\n",
          #"     <phi>   = ", af.mean(params.phi[N_g:-N_g, N_g:-N_g]), "\n",
          "     <n>     = ", af.mean(density[0, N_g:-N_g, N_g:-N_g]), "\n",
          "     max(n)  = ", af.max(density[0, N_g:-N_g, N_g:-N_g]), "\n",
          "     |E1|    = ", af.mean(af.abs(nls.cell_centered_EM_fields[0, N_g:-N_g, N_g:-N_g])),
          "\n",
          "     |E2|    = ", af.mean(af.abs(nls.cell_centered_EM_fields[1, N_g:-N_g, N_g:-N_g]))
         )
    print("--------------------\n")

    h5f = h5py.File('dumps/%06d'%(time_index+1) + '.h5', 'w')
    h5f.create_dataset('n', data = density[:, N_g:-N_g, N_g:-N_g])
    h5f.close()


