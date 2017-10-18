import arrayfire as af

from bolt.lib.nonlinear_solver.FVM_solver.riemann_solver \
    import riemann_solver

# Equation to solve:
# df/dt + d(C_q1 * f)/dq1 + d(C_q2 * f)/dq2 = C[f]
# Grid convention considered:

#                  (i, j+1/2)
#              X-------o-------X
#              |               |
#              |               |
#   (i-1/2, j) o       o       o (i+1/2, j)
#              |     (i, j)    |
#              |               |
#              X-------o-------X
#                  (i, j-1/2)

# Using the finite volume method:
# f_{i, j}/dt  = ((- (C_q1 * f)_{i + 1/2, j} + (C_q1 * f)_{i - 1/2, j})/dq1
#                 (- (C_q2 * f)_{i, j + 1/2} + (C_q2 * f)_{i, j - 1/2})/dq2
#                  +  C[f_{i, j}]
#                )

def df_dt(f, C_q1, C_q2, dq1, dq2, source):

    left_flux, right_flux, bot_flux, top_flux = riemann_solver(f, C_q1, C_q2)

    df_dt = - (right_flux - left_flux)/dq1
            - (top_flux   - bot_flux )/dq2
            + source(f) 

    af.eval(df_dt)
    return(df_dt)
 