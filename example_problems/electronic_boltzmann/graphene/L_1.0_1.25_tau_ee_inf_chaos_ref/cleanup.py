import os
import numpy as np

# Runs from tau = [5.8, 8.0] need to be run.

start_index = 1.91
end_index   = 1.991
step        = 0.01

current_filepath = \
        '/gpfs/u/home/NECD/NECDsdrs/scratch/bolt/example_problems/electronic_boltzmann/graphene/ref'
source_filepath = \
        '/gpfs/u/home/NECD/NECDsdrs/scratch/bolt/example_problems/electronic_boltzmann/graphene/ref'

for index in np.arange(start_index, end_index, step):
    filepath = \
    '/gpfs/u/home/NECD/NECDsdrs/scratch/bolt/example_problems/electronic_boltzmann/graphene/L_1.0_2.5_tau_ee_inf_tau_eph_%.2f_DC'%index

    os.system("cp " + source_filepath + "/dcs* " + filepath+"/.")
   
    # Schedule job after changing to run directory so that generated slurm file
    # is stored in that directory
    os.chdir(filepath)
    os.system("sbatch dcs_job_script")
    os.chdir(current_filepath) # Return to job firing script's directory

