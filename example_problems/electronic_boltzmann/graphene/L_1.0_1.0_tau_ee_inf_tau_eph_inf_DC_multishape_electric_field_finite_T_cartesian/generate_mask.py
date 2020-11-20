import numpy as np
from scipy.signal import convolve2d

import domain


dq1 = domain.dq1
dq2 = domain.dq2

N_g  = domain.N_ghost
N_q1 = domain.N_q1 + 2*N_g #Size of mask should include the ghost zones
N_q2 = domain.N_q2 + 2*N_g

def dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    
def make_circle(tiles, cx, cy, r, A):
    for x in range(cx - r, cx + r):
        for y in range(cy - r, cy + r):
            if dist(cx, cy, x, y) <= r:
                tiles[x, y] = -A

def electric_field():

    
    # Numpy array of mask (for phi)
    A = 0.030
    discrete_mask = np.zeros([N_q1, N_q2])

    ####### 
    
    cx = 40#N_q1 // 2
    cy = 65#N_q2 // 2
    r  = 15
    #cx = 70#N_q1 // 2
    #cy = 70#N_q2 // 2
    #r  = 30
    
    discrete_mask = np.zeros([N_q1, N_q2])
    make_circle(discrete_mask, cx, cy, r, A)
    
    #######
    
    center = int(N_q1/2)
    length = 70
    for i in range(N_q1):
        for j in range(N_q2):
            start_idx = 0
            end_idx   = 0
            start_i   = 90
            end_i     = 130
    
            if (j > start_i) and (j < end_i):            
                start_idx = center - int(length/2)
                end_idx   = start_idx + length
                length    = length - 2
                discrete_mask[start_idx:end_idx, j] = -A
                     
    #######
    
    off_center = 60
    length = 36
    for i in range(N_q1):
        start_i   = 80
        end_i     = 110
        if (i > start_i) and (i < end_i):            
            start_idx = off_center - int(length/2)
            end_idx   = start_idx + length
            length    = length - 2
            #print (i, length)
            discrete_mask[i, start_idx:end_idx] = -A
            
    off_center = 60
    length = 0
    for i in range(N_q1):
        start_i   = 62
        end_i     = 82
        if (i > start_i) and (i < end_i):            
            start_idx = off_center - int(length/2)
            end_idx   = start_idx + length
            length    = length + 2
            #print (i, length)
            discrete_mask[i, start_idx:end_idx] = -A
    
    # Smoothen mask
    filter_size    = 15
    spatial_filter = 1./filter_size**2.*np.ones([filter_size, filter_size])
    discrete_mask  = convolve2d(discrete_mask,
                            spatial_filter,
                            mode='same',
                            boundary='fill',
                            fillvalue=0. # not sure if fillvalue is working/how it works
                           )

    np.savetxt("mask.txt", discrete_mask)
    
    # Calculate the derivatives
    E_x = (np.roll(discrete_mask, -1, axis = 0) - np.roll(discrete_mask, 1, axis = 0))/(2*dq1)
    E_y = (np.roll(discrete_mask, -1, axis = 1) - np.roll(discrete_mask, 1, axis = 1))/(2*dq1)

    np.savetxt("E_x_1.txt", E_x)
    np.savetxt("E_y_1.txt", E_y)
    np.savetxt("E_x_2.txt", E_x)
    np.savetxt("E_y_2.txt", E_y)
    
    return

electric_field()
