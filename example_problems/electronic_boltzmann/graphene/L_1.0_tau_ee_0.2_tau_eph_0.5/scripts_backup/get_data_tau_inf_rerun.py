import os
import numpy as np

start_index = 0.96
end_index   = 0.961
step        = 0.02

current_filepath = \
        '/home/mchandra/cci_data'

for index in np.arange(start_index, end_index, step):

    source_filepath = \
    'cci:/gpfs/u/home/NECD/NECDsdrs/scratch/bolt/example_problems/electronic_boltzmann/graphene/L_1.0_%.2f_tau_ee_inf_tau_eph_inf_rerun/dumps'%index

    filepath = \
    '/home/mchandra/cci_data/dumps_ballistic_tau_inf_L_1.0_%.2f_rerun'%index

    # If folder does not exist, make one and add all files from source folder

    if not os.path.isdir(filepath):
        os.mkdir(filepath)

        os.system("scp " + source_filepath + "/moments_017*.h5 " + filepath+"/.")
        os.system("scp " + source_filepath + "/moments_017*.h5.info " + filepath+"/.")
        os.system("scp " + source_filepath + "/lagrange_multipliers_017*.h5 " + filepath+"/.")
        os.system("scp " + source_filepath + "/lagrange_multipliers_017*.h5.info " + filepath+"/.")
