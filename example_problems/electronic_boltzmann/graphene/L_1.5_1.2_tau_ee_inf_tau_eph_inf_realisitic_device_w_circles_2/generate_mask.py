import numpy as np
from scipy.signal import convolve2d
import pylab as pl

import domain

mpiprocs = 6

dq1 = domain.dq1
dq2 = domain.dq2

N_buffer =  10

N_g  = domain.N_ghost
N_q1 = domain.N_q1 + 2*N_g + 2*N_buffer #Size of mask should include the ghost zones
N_q2 = domain.N_q2 + 2*N_g

X = domain.q1_start  + (0.5 + np.arange(N_q1)) * (domain.q1_end - domain.q1_start)/N_q1
Y = domain.q2_start  + (0.5 + np.arange(N_q2)) * (domain.q2_end - domain.q2_start)/N_q2

X, Y = np.meshgrid(X, Y, indexing='ij')

def dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    
def make_circle(tiles, cx, cy, r, A):
    for x in range(cx - r, cx + r):
        for y in range(cy - r, cy + r):
            if dist(cx, cy, x, y) <= r:
                tiles[x, y] = -A

def electric_field():

    
    # Numpy array of mask (for phi)
    A = 0.060
    discrete_mask = np.zeros([N_q1, N_q2])

    indices_1 = (X < 0.25) & (Y < 0.5)
    indices_2 = (X < 0.25) & (Y > 0.7)
    indices_3 = (X > 1.25) & (Y < 0.5)
    indices_4 = (X > 1.25) & (Y > 0.7)
    indices_5 = (Y < 0.1)
    indices_6 = (Y > 1.1)
   
    print (indices_1.shape, X.shape) 
    discrete_mask[indices_1] = -A
    discrete_mask[indices_2] = -A
    discrete_mask[indices_3] = -A
    discrete_mask[indices_4] = -A
    discrete_mask[indices_5] = -A
    discrete_mask[indices_6] = -A
    
    filter_size    = 10
    spatial_filter = 1./filter_size**2.*np.ones([filter_size, filter_size])
    discrete_mask_1  = convolve2d(discrete_mask,
                                spatial_filter,
                                mode='same',
                                boundary='fill',
                                fillvalue=0. # not sure if fillvalue is working/how it works
                               )

 
   
    discrete_mask = np.zeros([N_q1, N_q2])
    
    r = 6
    
    make_circle(discrete_mask, int(N_buffer + 35*1.33), int(1.33*20), r, A)
    make_circle(discrete_mask, int(N_buffer + 35*1.33), int(1.33*40), r, A)
    make_circle(discrete_mask, int(N_buffer + 35*1.33), int(1.33*60), r, A)
    make_circle(discrete_mask, int(N_buffer + 35*1.33), int(1.33*80), r, A)
    make_circle(discrete_mask, int(N_buffer + 35*1.33), int(1.33*100),r, A)
    
    make_circle(discrete_mask, int(N_buffer + 55*1.33), int(1.33*20), r, A)
    make_circle(discrete_mask, int(N_buffer + 55*1.33), int(1.33*40), r, A)
    make_circle(discrete_mask, int(N_buffer + 55*1.33), int(1.33*60), r, A)
    make_circle(discrete_mask, int(N_buffer + 55*1.33), int(1.33*80), r, A)
    make_circle(discrete_mask, int(N_buffer + 55*1.33), int(1.33*100),r, A)
    
    make_circle(discrete_mask, int(N_buffer + 1.33*75), int(1.33*20), r, A)
    make_circle(discrete_mask, int(N_buffer + 1.33*75), int(1.33*40), r, A)
    make_circle(discrete_mask, int(N_buffer + 1.33*75), int(1.33*60), r, A)
    make_circle(discrete_mask, int(N_buffer + 1.33*75), int(1.33*80), r, A)
    make_circle(discrete_mask, int(N_buffer + 1.33*75), int(1.33*100),r, A)
    
    make_circle(discrete_mask, int(N_buffer + 1.33*95), int(1.33*20), r, A)
    make_circle(discrete_mask, int(N_buffer + 1.33*95), int(1.33*40), r, A)
    make_circle(discrete_mask, int(N_buffer + 1.33*95), int(1.33*60), r, A)
    make_circle(discrete_mask, int(N_buffer + 1.33*95), int(1.33*80), r, A)
    make_circle(discrete_mask, int(N_buffer + 1.33*95), int(1.33*100),r, A)
    
    make_circle(discrete_mask, int(N_buffer + 1.33*115), int(1.33*20), r, A)
    make_circle(discrete_mask, int(N_buffer + 1.33*115), int(1.33*40), r, A)
    make_circle(discrete_mask, int(N_buffer + 1.33*115), int(1.33*60), r, A)
    make_circle(discrete_mask, int(N_buffer + 1.33*115), int(1.33*80), r, A)
    make_circle(discrete_mask, int(N_buffer + 1.33*115), int(1.33*100),r, A)
    
    filter_size    = 5
    spatial_filter = 1./filter_size**2.*np.ones([filter_size, filter_size])
    discrete_mask_2  = convolve2d(discrete_mask,
                                spatial_filter,
                                mode='same',
                                boundary='fill',
                                fillvalue=0. # not sure if fillvalue is working/how it works
                               )

    final_mask = discrete_mask_1 + discrete_mask_2
    #final_mask = np.flip(final_mask, axis = 1)
    
    # Calculate the derivatives
    E_x = (np.roll(final_mask, -1, axis = 0) - np.roll(final_mask, 1, axis = 0))/(2*dq1)
    E_y = (np.roll(final_mask, -1, axis = 1) - np.roll(final_mask, 1, axis = 1))/(2*dq1)

    print("final :", final_mask.shape)
    np.savetxt("mask.txt", final_mask[N_buffer:-N_buffer])
    pl.contourf((final_mask[N_buffer:-N_buffer, :]).T, 100, cmap = 'bwr')
    pl.gca().set_aspect(1)
    pl.savefig("images/iv3.png")
    for i in range(mpiprocs):
        np.savetxt("fields/E_x_%d.txt"%i, E_x[N_buffer:-N_buffer])
        np.savetxt("fields/E_y_%d.txt"%i, E_y[N_buffer:-N_buffer])
    
    return

electric_field()
