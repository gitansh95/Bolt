import numpy as np
import arrayfire as af

import domain
import params

from bolt.lib.utils.coord_transformation import quadratic_test, quadratic


def get_cartesian_coords(q1, q2,
                         q1_start_local_left=None,
                         q2_start_local_bottom=None,
                         return_jacobian = False
                        ):

    q1_midpoint = 0.5*(af.max(q1) + af.min(q1))
    q2_midpoint = 0.5*(af.max(q2) + af.min(q2))

    d_q1          = (q1[0, 0, 1, 0] - q1[0, 0, 0, 0]).scalar()
    d_q2          = (q2[0, 0, 0, 1] - q2[0, 0, 0, 0]).scalar()

    N_g      = domain.N_ghost
    N_q1     = q1.dims()[2] - 2*N_g # Manually apply quadratic transformation for each zone along q1
    N_q2     = q2.dims()[3] - 2*N_g # Manually apply quadratic transformation for each zone along q1

    # Default initializsation to rectangular grid
    x = q1
    y = q2*(1 + q1)
    jacobian = None

    if (q1_start_local_left != None and q2_start_local_bottom != None):


        if (return_jacobian):
            return (x, y, jacobian)
        else:
            return(x, y)

    else:
        print("Error in get_cartesian_coords(): q1_start_local_left or q2_start_local_bottom not provided")


