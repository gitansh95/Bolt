#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import arrayfire as af

import coords

def apply_shearing_box_bcs_f(self, boundary):
    """
    Applies the shearing box boundary conditions along boundary specified 
    for the distribution function
    
    Parameters
    ----------
    boundary: str
              Boundary along which the boundary condition is to be applied.
    """

    N_g = self.N_ghost
    q     = self.physical_system.params.q 
    omega = self.physical_system.params.omega
    
    L_q1  = self.q1_end - self.q1_start
    L_q2  = self.q2_end - self.q2_start

    if(boundary == 'left'):
        sheared_coordinates = self.q2_center[:, :, :N_g] - q * omega * L_q1 * self.time_elapsed
        
        # Applying periodic boundary conditions to the points which are out of domain:
        while(af.sum(sheared_coordinates>self.q2_end) != 0):
            sheared_coordinates = af.select(sheared_coordinates>self.q2_end,
                                            sheared_coordinates - L_q2,
                                            sheared_coordinates
                                           )

        while(af.sum(sheared_coordinates<self.q2_start) != 0):
            sheared_coordinates = af.select(sheared_coordinates<self.q2_start,
                                            sheared_coordinates + L_q2,
                                            sheared_coordinates
                                           )

        # Reordering from (N_p, N_s, N_q1, N_q2) --> (N_q1, N_q2, N_p, N_s)
        # and reordering back from (N_q1, N_q2, N_p, N_s) --> (N_p, N_s, N_q1, N_q2)

        self.f[:, :, :N_g] = af.reorder(af.approx2(af.reorder(self.f[:, :, :N_g], 2, 3, 0, 1),
                                                     af.reorder(self.q1_center[:, :, :N_g], 2, 3, 0, 1),
                                                     af.reorder(sheared_coordinates, 2, 3, 0, 1),
                                                     af.INTERP.BICUBIC_SPLINE,
                                                     xp = af.reorder(self.q1_center[:, :, :N_g], 2, 3, 0, 1),
                                                     yp = af.reorder(self.q2_center[:, :, :N_g], 2, 3, 0, 1)
                                                    ),
                                          2, 3, 0, 1
                                         )
        
    elif(boundary == 'right'):
        sheared_coordinates = self.q2_center[:, :, -N_g:] + q * omega * L_q1 * self.time_elapsed

        # Applying periodic boundary conditions to the points which are out of domain:
        while(af.sum(sheared_coordinates>self.q2_end) != 0):
            sheared_coordinates = af.select(sheared_coordinates>self.q2_end,
                                            sheared_coordinates - L_q2,
                                            sheared_coordinates
                                           )

        while(af.sum(sheared_coordinates<self.q2_start) != 0):
            sheared_coordinates = af.select(sheared_coordinates<self.q2_start,
                                            sheared_coordinates + L_q2,
                                            sheared_coordinates
                                           )

        # Reordering from (N_p, N_s, N_q1, N_q2) --> (N_q1, N_q2, N_p, N_s)
        # and reordering back from (N_q1, N_q2, N_p, N_s) --> (N_p, N_s, N_q1, N_q2)

        self.f[:, :, -N_g:] = af.reorder(af.approx2(af.reorder(self.f[:, :, -N_g:], 2, 3, 0, 1),
                                                      af.reorder(self.q1_center[:, :, -N_g:], 2, 3, 0, 1),
                                                      af.reorder(sheared_coordinates, 2, 3, 0, 1),
                                                      af.INTERP.BICUBIC_SPLINE,
                                                      xp = af.reorder(self.q1_center[:, :, -N_g:], 2, 3, 0, 1),
                                                      yp = af.reorder(self.q2_center[:, :, -N_g:], 2, 3, 0, 1)
                                                     ),
                                            2, 3, 0, 1
                                           )

    elif(boundary == 'bottom'):

        sheared_coordinates = self.q1_center[:, :, :, :N_g] - q * omega * L_q2 * self.time_elapsed

        # Applying periodic boundary conditions to the points which are out of domain:
        while(af.sum(sheared_coordinates>self.q1_end) != 0):
            sheared_coordinates = af.select(sheared_coordinates>self.q1_end,
                                            sheared_coordinates - L_q1,
                                            sheared_coordinates
                                           )

        while(af.sum(sheared_coordinates<self.q1_start) != 0):
            sheared_coordinates = af.select(sheared_coordinates<self.q1_start,
                                            sheared_coordinates + L_q1,
                                            sheared_coordinates
                                           )

        # Reordering from (N_p, N_s, N_q1, N_q2) --> (N_q1, N_q2, N_p, N_s)
        # and reordering back from (N_q1, N_q2, N_p, N_s) --> (N_p, N_s, N_q1, N_q2)

        self.f[:, :, :, :N_g] = af.reorder(af.approx2(af.reorder(self.f[:, :, :, :N_g], 2, 3, 0, 1),
                                                        af.reorder(sheared_coordinates, 2, 3, 0, 1),
                                                        af.reorder(self.q2_center[:, :, :, :N_g], 2, 3, 0, 1),
                                                        af.INTERP.BICUBIC_SPLINE,
                                                        xp = af.reorder(self.q1_center[:, :, :, :N_g], 2, 3, 0, 1),
                                                        yp = af.reorder(self.q2_center[:, :, :, :N_g], 2, 3, 0, 1)
                                                       ),
                                             2, 3, 0, 1
                                            )

    elif(boundary == 'top'):

        sheared_coordinates = self.q1_center[:, :, :, -N_g:] + q * omega * L_q2 * self.time_elapsed

        # Applying periodic boundary conditions to the points which are out of domain:
        while(af.sum(sheared_coordinates>self.q1_end) != 0):
            sheared_coordinates = af.select(sheared_coordinates>self.q1_end,
                                            sheared_coordinates - L_q1,
                                            sheared_coordinates
                                           )

        while(af.sum(sheared_coordinates<self.q1_start) != 0):
            sheared_coordinates = af.select(sheared_coordinates<self.q1_start,
                                            sheared_coordinates + L_q1,
                                            sheared_coordinates
                                           )
        
        # Reordering from (N_p, N_s, N_q1, N_q2) --> (N_q1, N_q2, N_p, N_s)
        # and reordering back from (N_q1, N_q2, N_p, N_s) --> (N_p, N_s, N_q1, N_q2)

        self.f[:, :, :, -N_g:] = af.reorder(af.approx2(af.reorder(self.f[:, :, :, -N_g:], 2, 3, 0, 1),
                                                         af.reorder(sheared_coordinates, 2, 3, 0, 1),
                                                         af.reorder(self.q2_center[:, :, :, -N_g:], 2, 3, 0, 1),
                                                         af.INTERP.BICUBIC_SPLINE,
                                                         xp = af.reorder(self.q1_center[:, :, :, -N_g:], 2, 3, 0, 1),
                                                         yp = af.reorder(self.q2_center[:, :, :, -N_g:], 2, 3, 0, 1)
                                                        ),
                                              2, 3, 0, 1
                                             )

    else:
        raise Exception('Invalid choice for boundary')

    return

def apply_dirichlet_bcs_f(self, boundary):
    """
    Applies Dirichlet boundary conditions along boundary specified 
    for the distribution function
    
    Parameters
    ----------
    boundary: str
              Boundary along which the boundary condition is to be applied.
    """

    N_g = self.N_ghost
    
    if(self.physical_system.params.solver_method_in_q == 'FVM'):
        velocity_q1, velocity_q2 = \
            af.broadcast(self._C_q, self.time_elapsed, 
                         self.q1_center, self.q2_center,
                         self.p1_center, self.p2_center, self.p3_center,
                         self.physical_system.params
                        )

    else:
        velocity_q1, velocity_q2 = \
            af.broadcast(self._A_q, self.time_elapsed, 
                         self.q1_center, self.q2_center,
                         self.p1_center, self.p2_center, self.p3_center,
                         self.physical_system.params
                        )

    if(velocity_q1.elements() == self.N_species * self.N_p1 * self.N_p2 * self.N_p3):
        # If velocity_q1 is of shape (Np1 * Np2 * Np3)
        # We tile to get it to form (Np1 * Np2 * Np3, 1, Nq1, Nq2)
        velocity_q1 = af.tile(velocity_q1, 1, 1,
                              self.f.shape[2],
                              self.f.shape[3]
                             )

    if(velocity_q2.elements() == self.N_species * self.N_p1 * self.N_p2 * self.N_p3):
        # If velocity_q2 is of shape (Np1 * Np2 * Np3)
        # We tile to get it to form (Np1 * Np2 * Np3, 1, Nq1, Nq2)
        velocity_q2 = af.tile(velocity_q2, 1, 1,
                              self.f.shape[2],
                              self.f.shape[3]
                             )

    # Arguments that are passing to the called functions:
    args = (self.f, self.time_elapsed, self.q1_center, self.q2_center,
            self.p1_center, self.p2_center, self.p3_center, 
            self.physical_system.params
           )

    if(boundary == 'left'):
        f_left = self.boundary_conditions.f_left(*args)
        # Only changing inflowing characteristics:
        f_left = af.select(velocity_q1>0, f_left, self.f)
        self.f[:, :, :N_g] = f_left[:, :, :N_g]

    elif(boundary == 'right'):
        f_right = self.boundary_conditions.f_right(*args)
        # Only changing inflowing characteristics:
        f_right = af.select(velocity_q1<0, f_right, self.f)
        self.f[:, :, -N_g:] = f_right[:, :, -N_g:]

    elif(boundary == 'bottom'):
        f_bottom = self.boundary_conditions.f_bottom(*args)
        # Only changing inflowing characteristics:
        f_bottom = af.select(velocity_q2>0, f_bottom, self.f)
        self.f[:, :, :, :N_g] = f_bottom[:, :, :, :N_g]

    elif(boundary == 'top'):
        f_top = self.boundary_conditions.f_top(*args)
        # Only changing inflowing characteristics:
        f_top = af.select(velocity_q2<0, f_top, self.f)
        self.f[:, :, :, -N_g:] = f_top[:, :, :, -N_g:]

    else:
        raise Exception('Invalid choice for boundary')

    return

def apply_mirror_bcs_f_cartesian(self, boundary):
    """
    Applies mirror boundary conditions along boundary specified 
    for the distribution function when momentum space is on a cartesian grid
    
    Parameters
    ----------
    boundary: str
              Boundary along which the boundary condition is to be applied.
    """

    N_g = self.N_ghost

    if(boundary == 'left'):
        # x-0-x-0-x-0-|-0-x-0-x-0-x-....
        #   0   1   2   3   4   5
        # For mirror boundary conditions:
        # 0 = 5; 1 = 4; 2 = 3;
        self.f[:, :, :N_g] = af.flip(self.f[:, :, N_g:2 * N_g], 2)
        
        # The points in the ghost zone need to have direction 
        # of velocity reversed as compared to the physical zones 
        # they are mirroring. To do this we flip the axis that 
        # contains the variation in p1
        self.f[:, :, :N_g] = \
            self._convert_to_q_expanded(af.flip(self._convert_to_p_expanded(self.f), 
                                                0
                                               )
                                       )[:, :, :N_g]

    elif(boundary == 'right'):
        # ...-x-0-x-0-x-0-|-0-x-0-x-0-x
        #      -6  -5  -4  -3  -2  -1
        # For mirror boundary conditions:
        # -1 = -6; -2 = -5; -3 = -4;
        self.f[:, :, -N_g:] = af.flip(self.f[:, :, -2 * N_g:-N_g], 2)

        # The points in the ghost zone need to have direction 
        # of velocity reversed as compared to the physical zones 
        # they are mirroring. To do this we flip the axis that 
        # contains the variation in p1
        self.f[:, :, -N_g:] = \
            self._convert_to_q_expanded(af.flip(self._convert_to_p_expanded(self.f), 
                                                0
                                               )
                                       )[:, :, -N_g:]

    elif(boundary == 'bottom'):
        # x-0-x-0-x-0-|-0-x-0-x-0-x-....
        #   0   1   2   3   4   5
        # For mirror boundary conditions:
        # 0 = 5; 1 = 4; 2 = 3;
        self.f[:, :, :, :N_g] = af.flip(self.f[:, :, :, N_g:2 * N_g], 3)

        # The points in the ghost zone need to have direction 
        # of velocity reversed as compared to the physical zones 
        # they are mirroring. To do this we flip the axis that 
        # contains the variation in p2
        self.f[:, :, :, :N_g] = \
            self._convert_to_q_expanded(af.flip(self._convert_to_p_expanded(self.f), 
                                                1
                                               )
                                       )[:, :, :, :N_g]

    elif(boundary == 'top'):
        # ...-x-0-x-0-x-0-|-0-x-0-x-0-x
        #      -6  -5  -4  -3  -2  -1
        # For mirror boundary conditions:
        # -1 = -6; -2 = -5; -3 = -4;
        self.f[:, :, :, -N_g:] = af.flip(self.f[:, :, :, -2 * N_g:-N_g], 3)

        # The points in the ghost zone need to have direction 
        # of velocity reversed as compared to the physical zones 
        # they are mirroring. To do this we flip the axis that 
        # contains the variation in p2
        self.f[:, :, :, -N_g:] = \
            self._convert_to_q_expanded(af.flip(self._convert_to_p_expanded(self.f), 
                                                1
                                               )
                                       )[:, :, :, -N_g:]

    else:
        raise Exception('Invalid choice for boundary')

    return

def mirror_at_an_angle_polar2D(self, f, theta):
    """
    Applies mirror boundary conditions, with the mirror at an angle theta
    with the positive x-axis. Works with polar2D momentum space grid.
    
    For example, the vertical boundaries have theta = np.pi/2 and the 
    horizontal boundaries have theta = 0.

    Parameters
    -------------
    f     : array
            distribution function in q_expanded form
    theta : float
            Angle of the mirror boundary with respect to the positive x-axis
    """

    # theta_p = incident angle of particle wrt positive x-axis
    # theta   = angle of the mirror boundary wrt positive x-axis

    # In polar coordinates, the reflection of a particle moving at an angle
    # theta_p wrt the positive x-axis off a boundary which is at an angle
    # theta wrt the positive x-axis results in the reflected particle moving
    # at an angle 2*theta - theta_p wrt the positive x-axis.


    tmp = self._convert_to_p_expanded(f)
    # Operation to be performed : 2*theta - theta_p
    # We split this operation into 2 steps as shown below.

    # Operation 1 : theta_prime = theta_p - 2*theta
    # To do this, we shift the array along the axis that contains the variation in p_theta
    N_theta = self.N_p2
    no_of_shifts = int((theta/np.pi)*N_theta)
    
    print ("boundaries.py, no of shifts : ", no_of_shifts)

    # Shift along the p2 axis in the negative direction
    tmp = af.shift(tmp, d0 = 0, d1 = -no_of_shifts)

    # Operation 2 : theta_out = -theta_prime
    # To do this we flip the axis that contains the variation in p_theta
    tmp = af.flip(tmp, 1)

    tmp = self._convert_to_q_expanded(tmp)
    
    return(tmp)

def apply_mirror_bcs_f_polar2D(self, boundary):
    """
    Applies mirror boundary conditions along boundary specified 
    for the distribution function when momentum space is on a 2D polar grid
    
    Parameters
    ----------
    boundary: str
              Boundary along which the boundary condition is to be applied.
    """

    N_g = self.N_ghost
    theta_left   = self.physical_system.params.theta_left.to_ndarray()
    theta_right  = self.physical_system.params.theta_right.to_ndarray()
    theta_top    = self.physical_system.params.theta_top.to_ndarray()
    theta_bottom = self.physical_system.params.theta_bottom.to_ndarray()

   # print ("theta : ", theta)

    if(boundary == 'left'):
        
        # x-0-x-0-x-0-|-0-x-0-x-0-x-....
        #   0   1   2   3   4   5
        # For mirror boundary conditions:
        # 0 = 5; 1 = 4; 2 = 3;
        self.f[:, :, :N_g] = af.flip(self.f[:, :, N_g:2 * N_g], 2)
        
        index = 0
        for theta_boundary in theta_left[1:]:
            #print ("boundaries.py, theta_boundary left : ", theta_boundary)
            self.f[:, :, :N_g, index:index+1] = mirror_at_an_angle_polar2D(self, self.f, theta_boundary)[:, :, :N_g, index:index+1]
            index = index + 1

    elif(boundary == 'right'):
    
        # ...-x-0-x-0-x-0-|-0-x-0-x-0-x
        #      -6  -5  -4  -3  -2  -1
        # For mirror boundary conditions:
        # -1 = -6; -2 = -5; -3 = -4;
        self.f[:, :, -N_g:] = af.flip(self.f[:, :, -2 * N_g:-N_g], 2)

        index = 0
        for theta_boundary in theta_right[1:]:
            #print ("boundaries.py, theta_boundary right : ", theta_boundary)
            self.f[:, :, -N_g:, index:index+1] = mirror_at_an_angle_polar2D(self, self.f, theta_boundary)[:, :, -N_g:, index:index+1]
            index = index + 1

    elif(boundary == 'bottom'):

        # x-0-x-0-x-0-|-0-x-0-x-0-x-....
        #   0   1   2   3   4   5
        # For mirror boundary conditions:
        # 0 = 5; 1 = 4; 2 = 3;
        self.f[:, :, :, :N_g] = af.flip(self.f[:, :, :, N_g:2 * N_g], 3)
    
        index = 0
        for theta_boundary in theta_bottom[1:]:
           #print ("boundaries.py, theta_boundary bottom : ", theta_boundary)
           self.f[:, :, index:index+1, :N_g] = mirror_at_an_angle_polar2D(self, self.f, theta_boundary)[:, :, index:index+1, :N_g]
           index = index + 1

    elif(boundary == 'top'):

        # ...-x-0-x-0-x-0-|-0-x-0-x-0-x
        #      -6  -5  -4  -3  -2  -1
        # For mirror boundary conditions:
        # -1 = -6; -2 = -5; -3 = -4;
        self.f[:, :, :, -N_g:] = af.flip(self.f[:, :, :, -2 * N_g:-N_g], 3)

        index = 0
        for theta_boundary in theta_top[1:]:
            #print ("boundaries.py, theta_boundary top : ", theta_boundary)
            self.f[:, :, index:index+1, -N_g:] = mirror_at_an_angle_polar2D(self, self.f, theta_boundary)[:, :, index:index+1, -N_g:]
            index = index + 1

    else:
        raise Exception('Invalid choice for boundary')

    return

def apply_bcs_f(self):
    """
    Applies boundary conditions to the distribution function as specified by 
    the user in params.
    """


    if(self.performance_test_flag == True):
        tic = af.time()

    # Obtaining start coordinates for the local zone
    # Additionally, we also obtain the size of the local zone
    ((i_q1_start, i_q2_start), (N_q1_local, N_q2_local)) = self._da_f.getCorners()
    # Obtaining the end coordinates for the local zone
    (i_q1_end, i_q2_end) = (i_q1_start + N_q1_local - 1, i_q2_start + N_q2_local - 1)

    # If local zone includes the left physical boundary:
    if(i_q1_start == 0):

        if(self.boundary_conditions.in_q1_left == 'dirichlet'):
            apply_dirichlet_bcs_f(self, 'left')

        elif(self.boundary_conditions.in_q1_left == 'mirror'):
            if (self.physical_system.params.p_space_grid == 'cartesian'):
                apply_mirror_bcs_f_cartesian(self, 'left')            
            elif (self.physical_system.params.p_space_grid == 'polar2D'):
                apply_mirror_bcs_f_polar2D(self, 'left')
            else :
                raise NotImplementedError('Unsupported coordinate system in p_space')

        elif(self.boundary_conditions.in_q1_left == 'mirror+dirichlet'):
            if (self.physical_system.params.p_space_grid == 'cartesian'):
                apply_mirror_bcs_f_cartesian(self, 'left')
            elif (self.physical_system.params.p_space_grid == 'polar2D'):
                apply_mirror_bcs_f_polar2D(self, 'left')
            else :
                raise NotImplementedError('Unsupported coordinate system in p_space')
            apply_dirichlet_bcs_f(self, 'left')
        
        # This is automatically handled by the PETSc function globalToLocal()
        elif(   self.boundary_conditions.in_q1_left == 'periodic'
             or self.boundary_conditions.in_q1_left == 'none' # no ghost zones (1D)
            ):
            pass

        elif(self.boundary_conditions.in_q1_left == 'shearing-box'):
            apply_shearing_box_bcs_f(self, 'left')

        else:
            raise NotImplementedError('Unavailable/Invalid boundary condition')
    
    # If local zone includes the right physical boundary:
    if(i_q1_end == self.N_q1 - 1):

        if(self.boundary_conditions.in_q1_right == 'dirichlet'):
            apply_dirichlet_bcs_f(self, 'right')

        elif(self.boundary_conditions.in_q1_right == 'mirror'):
            if (self.physical_system.params.p_space_grid == 'cartesian'):
                apply_mirror_bcs_f_cartesian(self, 'right')
            elif (self.physical_system.params.p_space_grid == 'polar2D'):
                apply_mirror_bcs_f_polar2D(self, 'right')
            else:
                raise NotImplementedError('Unsupported coordinate system in p_space')
        
        elif(self.boundary_conditions.in_q1_right == 'mirror+dirichlet'):
            if (self.physical_system.params.p_space_grid == 'cartesian'):
                apply_mirror_bcs_f_cartesian(self, 'right')
            elif (self.physical_system.params.p_space_grid == 'polar2D'):
                apply_mirror_bcs_f_polar2D(self, 'right')
            else:
                raise NotImplementedError('Unsupported coordinate system in p_space')
            apply_dirichlet_bcs_f(self, 'right')

        # This is automatically handled by the PETSc function globalToLocal()
        elif(   self.boundary_conditions.in_q1_right == 'periodic'
             or self.boundary_conditions.in_q1_right == 'none' # no ghost zones (1D)
            ):
            pass

        elif(self.boundary_conditions.in_q1_right == 'shearing-box'):
            apply_shearing_box_bcs_f(self, 'right')

        else:
            raise NotImplementedError('Unavailable/Invalid boundary condition')

    # If local zone includes the bottom physical boundary:
    if(i_q2_start == 0):

        if(self.boundary_conditions.in_q2_bottom == 'dirichlet'):
            apply_dirichlet_bcs_f(self, 'bottom')

        elif(self.boundary_conditions.in_q2_bottom == 'mirror'):
            if (self.physical_system.params.p_space_grid =='cartesian'):
                apply_mirror_bcs_f_cartesian(self, 'bottom')
            elif (self.physical_system.params.p_space_grid == 'polar2D'):
                apply_mirror_bcs_f_polar2D(self, 'bottom')
            else:
                raise NotImplementedError('Unsupported coordinate system in p_space')

        elif(self.boundary_conditions.in_q2_bottom == 'mirror+dirichlet'):
            if (self.physical_system.params.p_space_grid == 'cartesian'):
                apply_mirror_bcs_f_cartesian(self, 'bottom')
            elif (self.physical_system.params.p_space_grid == 'polar2D'):
                apply_mirror_bcs_f_polar2D(self, 'bottom')
            else:
                raise NotImplementedError('Unsupported coordinate system in p_space')
            apply_dirichlet_bcs_f(self, 'bottom')
        
        # This is automatically handled by the PETSc function globalToLocal()
        elif(   self.boundary_conditions.in_q2_bottom == 'periodic'
             or self.boundary_conditions.in_q2_bottom == 'none' # no ghost zones (1D)
            ):
            pass

        elif(self.boundary_conditions.in_q2_bottom == 'shearing-box'):
            apply_shearing_box_bcs_f(self, 'bottom')

        else:
            raise NotImplementedError('Unavailable/Invalid boundary condition')

    # If local zone includes the top physical boundary:
    if(i_q2_end == self.N_q2 - 1):

        if(self.boundary_conditions.in_q2_top == 'dirichlet'):
            apply_dirichlet_bcs_f(self, 'top')

        elif(self.boundary_conditions.in_q2_top == 'mirror'):
            if (self.physical_system.params.p_space_grid == 'cartesian'):
                apply_mirror_bcs_f_cartesian(self, 'top')
            elif (self.physical_system.params.p_space_grid == 'polar2D'):
                apply_mirror_bcs_f_polar2D(self, 'top')
            else:
                raise NotImplementedError('Unsupported coordinate system in p_space')
        
        elif(self.boundary_conditions.in_q2_top == 'mirror+dirichlet'):
            if (self.physical_system.params.p_space_grid == 'cartesian'):
                apply_mirror_bcs_f_cartesian(self, 'top')            
            elif (self.physical_system.params.p_space_grid == 'polar2D'):
                apply_mirror_bcs_f_polar2D(self, 'top')
            else:
                raise NotImplementedError('Unsupported coordinate system in p_space')
            apply_dirichlet_bcs_f(self, 'top')
        
        # This is automatically handled by the PETSc function globalToLocal()
        elif(   self.boundary_conditions.in_q2_top == 'periodic'
             or self.boundary_conditions.in_q2_top == 'none' # no ghost zones (1D)
            ):
            pass

        elif(self.boundary_conditions.in_q2_top == 'shearing-box'):
            apply_shearing_box_bcs_f(self, 'top')

        else:
            raise NotImplementedError('Unavailable/Invalid boundary condition')

    af.eval(self.f)

    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_apply_bcs_f += toc - tic
   
    return

