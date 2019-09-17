import os
import numpy as np

start_index = 0.220
end_index   = 0.2791
step        = 0.001

current_filepath = \
        '/gpfs/u/home/NECD/NECDsdrs/scratch/bolt/example_problems/electronic_boltzmann/graphene/ref_L_1.0_2.5_tau_ee_inf'
source_filepath = \
        '/gpfs/u/home/NECD/NECDsdrs/scratch/bolt/example_problems/electronic_boltzmann/graphene/ref_L_1.0_2.5_tau_ee_inf'

for index in np.arange(start_index, end_index, step):
    filepath = \
    '/gpfs/u/home/NECD/NECDsdrs/scratch/bolt/example_problems/electronic_boltzmann/graphene/L_1.0_2.5_tau_ee_inf_tau_eph_%.6f_DC_hr'%index

    # If folder does not exist, make one and add all files from source folder

    if not os.path.isdir(filepath):
        os.mkdir(filepath)
        os.mkdir(filepath+"/dumps")
        #os.mkdir(filepath+"/images")

        os.system("cp " + source_filepath + "/* " + filepath+"/.")
   
        # Change required files
        # Code copied from here : 
        # https://stackoverflow.com/questions/4454298/prepend-a-line-to-an-existing-file-in-python
        
        f = open(filepath + "/params.py", "r")
        old = f.read() # read everything in the file
        f.close() 
        
        f = open(filepath + "/params.py", "w")
        f.write("tau_mr = " + str(index) + " \n")
        f.write(old)
        f.close()

    # Schedule job after changing to run directory so that generated slurm file
    # is stored in that directory
    os.chdir(filepath)
    os.system("sbatch dcs_job_script")
    os.chdir(current_filepath) # Return to job firing script's directory

