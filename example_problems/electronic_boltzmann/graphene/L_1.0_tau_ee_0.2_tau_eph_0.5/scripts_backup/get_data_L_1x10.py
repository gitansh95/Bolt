import os
import numpy as np

start_index = 0.7
end_index   = 0.71
step        = 0.1

current_filepath = \
        '/home/mchandra/cci_data'

for index in np.arange(start_index, end_index, step):

    source_filepath = \
    'cci:/gpfs/u/home/NECD/NECDsdrs/scratch/bolt/example_problems/electronic_boltzmann/graphene/L_1.0_20.0_tau_ee_inf_tau_eph_%.1f/dumps'%index

    filepath = \
    '/home/mchandra/cci_data/dumps_L_1.0_20.0_tau_eph_%.1f'%index

    # If folder does not exist, make one and add all files from source folder

    if not os.path.isdir(filepath):
        os.mkdir(filepath)

        os.system("scp " + source_filepath + "/moments_009*.h5 " + filepath+"/.")
        os.system("scp " + source_filepath + "/moments_009*.h5.info " + filepath+"/.")
        os.system("scp " + source_filepath + "/lagrange_multipliers_009*.h5 " + filepath+"/.")
        os.system("scp " + source_filepath + "/lagrange_multipliers_009*.h5.info " + filepath+"/.")
