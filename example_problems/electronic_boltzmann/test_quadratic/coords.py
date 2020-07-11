import numpy as np
import arrayfire as af

import domain
import params

from bolt.lib.utils.coord_transformation import quadratic_test

def get_cartesian_coords(q1, q2, 
                         q1_start_local_left=None, 
                         q2_start_local_bottom=None,
                         return_jacobian = False
                        ):

    q1_midpoint = 0.5*(af.max(q1) + af.min(q1))
    q2_midpoint = 0.5*(af.max(q2) + af.min(q2))

    N_g = domain.N_ghost

    # Default initialisation to rectangular grid
    x = q1
    y = q2
    jacobian = [[1. + 0.*q1,      0.*q1],
                [     0.*q1, 1. + 0.*q1]
               ]

    # Radius and center of circular region
    radius          = 0.5
    center          = [0, 0]
    print ("dq1 : ", domain.dq1)
    print ("dq2 : ", domain.dq2)

    if (q1_start_local_left != None and q2_start_local_bottom != None):
        
        N_q1     = q1.dims()[2] - 2*N_g # Manually apply quadratic transformation for each zone along q1
        N_q2     = q2.dims()[3] - 2*N_g # Manually apply quadratic transformation for each zone along q1
        N        = 10#N_q1 
        x_0     = -radius/np.sqrt(2)

        # Initialize to zero
        x = 0*q1
        y = 0*q2

        # Loop over each zone in x
        for i in range(N_g, N_q1 + N_g):

            index = i - N_g # Index of the vertical slice, left-most being 0

            # Compute the x, y points using which the transformation will be defined
            # x, y nodes remain the same for each point on a vertical slice
            x_n           = x_0 + np.sqrt(2)*radius*index/N # Bottom-left
            y_n           = np.sqrt(radius**2 - x_n**2)
            
            print ("x_n, y_n : ", x_n, y_n)

            x_n_plus_1    = x_0 + np.sqrt(2)*radius*(index+1)/N # Bottom-right
            y_n_plus_1    = np.sqrt(radius**2 - x_n_plus_1**2)

            x_n_plus_half = x_0 + np.sqrt(2)*radius*(index+0.5)/N # Bottom-center
            y_n_plus_half = np.sqrt(radius**2 - x_n_plus_half**2)


            x_y_bottom_left   = [x_n,           y_n]
            x_y_bottom_center = [x_n_plus_half, y_n_plus_half]
            x_y_bottom_right  = [x_n_plus_1,    y_n_plus_1]
    
            x_y_left_center   = [x_n,        (1+y_n)/2]
            x_y_right_center  = [x_n_plus_1, (1+y_n_plus_1)/2]
    
            x_y_top_left      = [x_n,           1]
            x_y_top_center    = [x_n_plus_half, 1]
            x_y_top_right     = [x_n_plus_1,    1]

            # TODO : Testing
            if (i == N_g):
                save_array = x_y_bottom_left.copy()
                save_array.extend(x_y_bottom_center)
                save_array.extend(x_y_bottom_right)
                save_array.extend(x_y_left_center)
                save_array.extend(x_y_right_center)
                save_array.extend(x_y_top_left)
                save_array.extend(x_y_top_center)
                save_array.extend(x_y_top_right)

                save_array = np.array(save_array)

                np.savetxt("input_to_quadratic_left_most_slice.txt", save_array)

            for j in range(N_g, N_q2 + N_g):

                print ("i, j, index : ", i, j, index)

                # Get the transformation (x_i, y_i) for each point (q1_i, q2_i) 
                q1_i = q1[0, 0, i, j]
                q2_i = q2[0, 0, i, j]

                x_i, y_i, jacobian_i = quadratic_test(q1_i, q2_i,
                                   x_y_bottom_left,   x_y_bottom_right,
                                   x_y_top_right,     x_y_top_left,
                                   x_y_bottom_center, x_y_right_center,
                                   x_y_top_center,    x_y_left_center,
                                   q1_start_local_left,
                                   q2_start_local_bottom,
                                  )

                
                print ("x_i, y_i : ", x_i.scalar(), y_i.scalar())


                # Reconstruct the x,y grid from the loop
                x[0, 0, i, j] = x_i
                y[0, 0, i, j] = y_i

                # TODO : Reconstruct jacobian
                #[[dx_dq1_i, dx_dq2_i], [dy_dq1_i, dy_dq2_i]] = jacobian_i
                #print ("dx_dq1_i : ", dx_dq1_i.dims())
                #jacobian =  TODO

               
        if (return_jacobian):
            return(x, y, jacobian)
        else:
            return(x, y)

    else:
        print("Error in get_cartesian_coords(): q1_start_local_left or q2_start_local_bottom not provided")
