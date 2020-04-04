import numpy as np
import arrayfire as af

# Time parameters:
dt      = 0.025/4 # ps
t_final = 10.     # ps


# Set to zero for no file-writing
dt_dump_f       = 2*dt #ps
# ALWAYS set dump moments and dump fields at same frequency:
dt_dump_moments = dt_dump_fields = 2*dt #ps


# Dimensionality considered in velocity space:
p_dim = 1
p_space_grid = 'polar2D' # Supports 'cartesian' or 'polar2D' grids
# Set p-space start and end points accordingly in domain.py

# Number of devices(GPUs/Accelerators) on each node:
num_devices = 6

# Spatial quantities (will be initialized to shape = [q1, q2] in initalize.py)
f_1         = None # Distribution function of the 1st domain
f_2         = None # Distribution function of the 2nd domain

initial_temperature = 12e-4
initial_mu          = 0.015

# Restart(Set to zero for no-restart):
latest_restart = True
t_restart = 0

