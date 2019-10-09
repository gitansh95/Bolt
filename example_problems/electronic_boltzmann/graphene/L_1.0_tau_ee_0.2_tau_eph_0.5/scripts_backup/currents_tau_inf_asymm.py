import numpy as np
import glob
import os
import PetscBinaryIO

#import domain

heights = np.arange(2.6, 3.01, 0.1)
#heights   = np.arange(2.5, 19.51, 0.5)
#heights   = np.append(heights_1, heights)

io = PetscBinaryIO.PetscBinaryIO()

print("Loading data...")

for l in heights:
    #filepath = os.getcwd() + "/dumps"
    filepath = \
	    "dumps_ballistic_tau_inf_L_1.0_%.2f_asymm"%(l)
    moment_files = np.sort(glob.glob(filepath+'/moment*.h5'))
    lagrange_multiplier_files = np.sort(glob.glob(filepath+'/lagrange_multiplier*.h5'))
    print (moment_files.size)
    print("dumps folder = ", filepath)

    N_q1 = 72
    N_q2 = int(round(N_q1*(l/2)))

    print ('N_q2 : ', N_q2)

    q1_start = 0.0
    q2_start = 0.0
    q1_end = 1.0
    q2_end = l/2

    print ("q2_end : ", q2_end)

    q1 = q1_start + (0.5 + np.arange(N_q1)) * (q1_end - q1_start)/N_q1
    q2 = q2_start + (0.5 + np.arange(N_q2)) * (q2_end - q2_start)/N_q2

    indices_local    = q2 < 0.25
    indices_nonlocal = (q2 > (q2_end-0.25)) & (q2 < q2_end)

    #print ('q2 size', q2.size)
    #print ("local indices : ", indices_nonlocal)
    
    voltage_local    = []
    voltage_nonlocal = []
    voltage_left     = []
    voltage_right    = []
    
    file_number = -2
    dump_file = moment_files[file_number]

    print("File number = ", file_number, ' of ', moment_files.size)

    left_edge = 0; right_edge = -1 
    
    moments = io.readBinaryFile(dump_file)
    moments = moments[0].reshape(N_q2, N_q1, 3)
    density = moments[:, :, 0]
    j_x     = moments[:, :, 1]
    j_y     = moments[:, :, 2]

    lagrange_multipliers_file = lagrange_multiplier_files[file_number]
    lagrange_multipliers = io.readBinaryFile(lagrange_multipliers_file)
    lagrange_multipliers = lagrange_multipliers[0].reshape(N_q2, N_q1, 7)

    mu    = lagrange_multipliers[:, :, 0]
    mu_ee = lagrange_multipliers[:, :, 1]
    T_ee  = lagrange_multipliers[:, :, 2]
    vel_x = lagrange_multipliers[:, :, 3]
    vel_y = lagrange_multipliers[:, :, 4]
    j_x_2 = lagrange_multipliers[:, :, 5]
    j_y_2 = lagrange_multipliers[:, :, 6]


    positive_lead = density[:, left_edge]
    negative_lead = density[:, right_edge]

    voltage_left.append(positive_lead)
    voltage_right.append(negative_lead)
    
    print ("J_x_2 : ", j_x_2.shape)
    current = j_x_2[:, left_edge]

    voltage_left = np.array(voltage_left)
    voltage_right = np.array(voltage_right)
    #current = np.array(current)

    #np.savez_compressed("edge_data/edge_L_1.0_%.2f_tau_ee_inf_tau_eph_inf.txt"%(l/2), left=voltage_left, right=voltage_right)
    #np.savetxt("data/voltage_left_L_1.0_2.5_tau_ee_inf_tau_eph_%.2f.txt"%(l),   voltage_left)
    #np.savetxt("data/voltage_right_L_1.0_2.5_tau_ee_inf_tau_eph_%.2f.txt"%(l), voltage_right)
    np.savetxt("edge_data/current_L_1.0_%.2f_tau_ee_inf_tau_eph_inf_asymm.txt"%(l/2), current)
