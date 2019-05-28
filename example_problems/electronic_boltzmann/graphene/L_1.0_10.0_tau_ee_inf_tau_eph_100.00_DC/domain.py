import numpy as np
import params

q1_start = 0.
q1_end   = 1.0
N_q1     = 72

q2_start = 0.
q2_end   = 10.0
N_q2     = 720

# If N_p1 > 1, mirror boundary conditions require p1 to be
# symmetric about zero

# In the polar representation of momentum space,
# p1 = p_r (magnitude of momentum)
# Here, p1_start and p1_end have been adjusted such that
# p_r_center is 1.0

## TODO: REMOVE HARDCODED VALUES, FIGURE OUT HOW TO IMPORT LIBRARIES IN DOMAIN
p1_start = 0.015 - 64.*params.boltzmann_constant*params.initial_temperature
p1_end   = 0.015 + 64.*params.boltzmann_constant*params.initial_temperature
N_p1     = 64

# If N_p2 > 1, mirror boundary conditions require p2 to be
# symmetric about zero

# In the polar representation of momentum space,
# p2 = p_theta (angle of momentum)
# N_p_theta MUST be even.

p2_start =  -np.pi
p2_end   =  np.pi
N_p2     =  512

# If N_p3 > 1, mirror boundary conditions require p3 to be
# symmetric about zero
p3_start = -0.5
p3_end   =  0.5
N_p3     =  1

N_ghost = 2

