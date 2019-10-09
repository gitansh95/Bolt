import os
import glob
import numpy as np


remove_fields     = True
remove_lagrangian = False
remove_moments    = False
remove_f          = True

start_index = 1
end_index   = 101
step        = 1

# Progress bar code copied from :
# https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1,
        length = 100, fill = '#'):
    """
    Call in a loop to create terminal progress bar
    @params:
    iteration   - Required  : current iteration (Int)
    total       - Required  : total iterations (Int)
    prefix      - Optional  : prefix  string (Str)
    suffix      - Optional  : suffix string (Str)
    decimals    - Optional  : positive number of decimals in percent complete (Int)
    length      - Optional  : character length of bar (Int)
    fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration/float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
       print()

for index in np.arange(start_index, end_index, step):
    filepath = \
    '/home/mchandra/gitansh/bolt/example_problems/electronic_boltzmann/graphene/L_1.0_2.5_tau_ee_inf_tau_eph_5.0_freq_%d/dumps'%index
    if remove_fields:
        os.system("rm " + filepath + "/fields_*")
    if remove_lagrangian:
        os.system("rm " + filepath + "/lagrangian_multipliers_*")
    if remove_moments:
        os.system("rm " + filepath + "/moments_*")
    if remove_f:
        os.system("rm " + filepath + "/f_*")
    printProgressBar(index + 1, end_index - start_index , prefix = 'Progress:', suffix = 'Complete', length = 50)

