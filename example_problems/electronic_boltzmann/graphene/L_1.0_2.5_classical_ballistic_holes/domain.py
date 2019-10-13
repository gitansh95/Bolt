q1_start = 0.
q1_end   = 1.0
N_q1     = 18

q2_start = 0.
q2_end   = 2.5
N_q2     = 45

# If N_p1 > 1, mirror boundary conditions require p1 to be
# symmetric about zero

# In the polar representation of momentum space,
# p1 = p_r (magnitude of momentum)
# Here, p1_start and p1_end have been adjusted such that
# p_r_center is 1.0

p1_start =  0.5
p1_end   =  1.5
N_p1     =  1

# If N_p2 > 1, mirror boundary conditions require p2 to be
# symmetric about zero

# In the polar representation of momentum space,
# p2 = p_theta (angle of momentum)
# N_p_theta MUST be even.

p2_start =  -3.14159265359
p2_end   =  3.14159265359
N_p2     =  8192

# If N_p3 > 1, mirror boundary conditions require p3 to be
# symmetric about zero
p3_start = -0.5
p3_end   =  0.5
N_p3     =  1

N_ghost = 2
