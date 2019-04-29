import os
import numpy as np

start_index = 50.0
end_index   = 97.01
step        = 1.0

current_filepath = \
        '/home/mchandra/gitansh/gitansh_bolt/example_problems/electronic_boltzmann/tmp'
source_filepath = \
        '/home/mchandra/gitansh/gitansh_bolt/example_problems/electronic_boltzmann/tmp'

for index in np.arange(start_index, end_index, step):
    filepath = \
    '/home/mchandra/gitansh/gitansh_bolt/example_problems/electronic_boltzmann/L_1.0_2.5_tau_ee_%.2f_tau_eph_100.00_DC'%index

    # If folder does not exist, make one and add all files from source folder

    if not os.path.isdir(filepath):
        os.mkdir(filepath)
        os.mkdir(filepath+"/dumps")
        os.mkdir(filepath+"/images")

        os.system("cp " + source_filepath + "/* " + filepath+"/.")
   
        # Change required files
        # Code copied from here : 
        # https://stackoverflow.com/questions/4454298/prepend-a-line-to-an-existing-file-in-python
        
        f = open(filepath + "/params.py", "r")
        old = f.read() # read everything in the file
        f.close() 
        
        f = open(filepath + "/params.py", "w")
        f.write("tau_mc = " + str(index) + " \n")
        f.write(old)
        f.close()

    # Schedule job after changing to run directory so that generated slurm file
    # is stored in that directory
    os.chdir(filepath)
    os.system("sbatch job_script")
    os.chdir(current_filepath) # Return to job firing script's directory

