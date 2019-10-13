import os
import numpy as np
import glob

start_index = 10.0
end_index   = 1000.0
step        = 1.0

current_filepath = \
        '/home/mchandra/cci_data'

for index in np.arange(start_index, end_index, step):

    print ("Index", index)

    source_filepath = \
        current_filepath + '/dumps_wire_cap_tau_%.2f'%index

    filepath = \
    '/home/mchandra/cci_data/dumps_wire_cap'%index

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
            "/moments_L_1.0_0.30_tau_ee_inf_tau_eph_%.2f.h5"%(index))
    os.system("cp " + lagrange + " " + filepath + \
            "/lagrange_multipliers_L_1.0_0.30_tau_ee_inf_tau_eph_%.2f.h5"%(index))
    os.system("cp " + moment_info + " " + filepath +
            "/moments_L_1.0_0.30_tau_ee_inf_tau_eph_%.2f.h5.info"%(index))
    os.system("cp " + lagrange_info + " " + filepath +
            "/lagrange_multipliers_L_1.0_0.30_tau_ee_inf_tau_eph_%.2f.h5.info"%(index))
