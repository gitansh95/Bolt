import numpy as np
from scipy.signal import convolve2d

import domain


dq1 = domain.dq1
dq2 = domain.dq2

N_buffer = 10
N_g  = domain.N_ghost
N_q1 = domain.N_q1 + 2*N_g + 2*N_buffer #Size of mask should include the ghost zones
N_q2 = domain.N_q2 + 2*N_g + 2*N_buffer

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

    discrete_mask[30+N_buffer:60+N_buffer, 30+N_buffer:60+N_buffer] = -A
 
    # Smoothen mask
    filter_size    = 7
    spatial_filter = 1./filter_size**2.*np.ones([filter_size, filter_size])
    discrete_mask  = convolve2d(discrete_mask,
                            spatial_filter,
                            mode='same',
                            boundary='fill',
                            fillvalue=0. # not sure if fillvalue is working/how it works
                           )

    np.savetxt("mask.txt", discrete_mask[N_buffer:-N_buffer, N_buffer:-N_buffer])
    
    # Calculate the derivatives
    E_x = (np.roll(discrete_mask, -1, axis = 0) - np.roll(discrete_mask, 1, axis = 0))/(2*dq1)
    E_y = (np.roll(discrete_mask, -1, axis = 1) - np.roll(discrete_mask, 1, axis = 1))/(2*dq1)

    mpiprocs = 6
    for i in range(mpiprocs):
        np.savetxt("fields/E_x_%d.txt"%i, E_x[N_buffer:-N_buffer, N_buffer:-N_buffer])
        np.savetxt("fields/E_y_%d.txt"%i, E_y[N_buffer:-N_buffer, N_buffer:-N_buffer])
    
    return

electric_field()
