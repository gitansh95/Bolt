import os
import numpy as np
import glob

start_index = 0.92
end_index   = 0.981
step        = 0.02

current_filepath = \
        '/home/mchandra/cci_data'

for index in np.arange(start_index, end_index, step):

    print ("Index", index)

    source_filepath = \
        current_filepath + '/dumps_ballistic_tau_inf_L_1.0_%.2f_rerun'%index

    filepath = \
    '/home/mchandra/cci_data/dumps_ballistic_tau_inf'%index

    moment_files 		  = np.sort(glob.glob(source_filepath+'/moment*.h5'))
    lagrange_multiplier_files = \
            np.sort(glob.glob(source_filepath+'/lagrange_multipliers*.h5'))
    moment_info_files 		  = \
            np.sort(glob.glob(source_filepath+'/moment*.h5.info'))
    lagrange_multiplier_info_files = \
            np.sort(glob.glob(source_filepath+'/lagrange_multipliers*.h5.info'))

    moment = moment_files[-2]
    moment_info = moment_info_files[-2]
    lagrange = lagrange_multiplier_files[-2]
    lagrange_info = lagrange_multiplier_files[-2]

    print (moment)
    print (lagrange)

    os.system("cp " + moment + " " + filepath + \
            "/moments_L_1.0_%.2f_tau_ee_inf_tau_eph_inf_rerun.h5"%(index/2))
    os.system("cp " + lagrange + " " + filepath + \
            "/lagrange_multiplers_L_1.0_%.2f_tau_ee_inf_tau_eph_inf_rerun.h5"%(index/2))
    os.system("cp " + moment_info + " " + filepath +
            "/moments_L_1.0_%.2f_tau_ee_inf_tau_eph_inf_rerun.h5.info"%(index/2))
    os.system("cp " + lagrange_info + " " + filepath +
            "/lagrange_multipliers_L_1.0_%.2f_tau_ee_inf_tau_eph_inf_rerun.info"%(index/2))
