import os
import numpy as np

start_index = 0.03
end_index   = 0.991
step        = 0.01

current_filepath = \
        '/home/mchandra/cci_data'

for index in np.arange(start_index, end_index, step):

    source_filepath = \
    'cci:/gpfs/u/home/NECD/NECDsdrs/scratch/bolt/example_problems/electronic_boltzmann/graphene/L_1.000_2.500_tau_ee_inf_tau_eph_%.2f_vel_1e-4/dumps'%index

    filepath = \
    '/home/mchandra/cci_data/L_1.000_2.5000_tau_ee_inf_tau_%.2f_vel_1e-4'%index

    # If folder does not exist, make one and add all files from source folder

    if not os.path.isdir(filepath):
        os.mkdir(filepath)

        os.system("scp " + source_filepath + "/moments_00*.h5 " + filepath+"/.")
        os.system("scp " + source_filepath + "/moments_00*.h5.info " + filepath+"/.")
        os.system("scp " + source_filepath + "/lagrange_multipliers_00*.h5 " + filepath+"/.")
        os.system("scp " + source_filepath + "/lagrange_multipliers_00*.h5.info " + filepath+"/.")
