import os
import numpy as np

start_index = 10.3
end_index   = 10.41
step        = 0.1

current_filepath = \
        '/home/mchandra/cci_data'

for index in np.arange(start_index, end_index, step):

    source_filepath = \
    'cci:/gpfs/u/home/NECD/NECDsdrs/scratch/bolt/example_problems/electronic_boltzmann/graphene/L_1.0_%.1f_tau_ee_inf_tau_eph_inf/dumps'%index

    filepath = \
    '/home/mchandra/cci_data/dumps_ballistic_tau_inf_L_1.0_%.1f'%index

    # If folder does not exist, make one and add all files from source folder

    if not os.path.isdir(filepath):
        os.mkdir(filepath)

        os.system("scp " + source_filepath + "/moments_016*.h5 " + filepath+"/.")
        os.system("scp " + source_filepath + "/moments_016*.h5.info " + filepath+"/.")
        os.system("scp " + source_filepath + "/lagrange_multipliers_016*.h5 " + filepath+"/.")
        os.system("scp " + source_filepath + "/lagrange_multipliers_016*.h5.info " + filepath+"/.")
