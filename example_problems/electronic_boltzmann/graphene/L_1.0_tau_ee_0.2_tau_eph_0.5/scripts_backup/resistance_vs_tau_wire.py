import arrayfire as af
import numpy as np
import glob
import yt
yt.enable_parallelism()
import PetscBinaryIO

N_q1 = 72
N_q2 = 18


q1_start = 0.0
q2_start = 0.0
q1_end = 1.0
q2_end = 0.25

#print ("q2_end : ", q2_end)

q1 = q1_start + (0.5 + np.arange(N_q1)) * (q1_end - q1_start)/N_q1
q2 = q2_start + (0.5 + np.arange(N_q2)) * (q2_end - q2_start)/N_q2

q2_meshgrid, q1_meshgrid = np.meshgrid(q2, q1)

source_start = 0.0
source_end   = 0.25

drain_start  = 0.0
drain_end    = 0.25

source_indices =  (q2 > source_start) & (q2 < source_end)
drain_indices  =  (q2 > drain_start)  & (q2 < drain_end )

# Left needs to be near source, right sensor near drain
sensor_1_left_start = source_start # um
sensor_1_left_end   = source_end # um

sensor_1_right_start = drain_start # um
sensor_1_right_end   = drain_end # um

sensor_1_left_indices  = (q2 > sensor_1_left_start ) & (q2 < sensor_1_left_end)
sensor_1_right_indices = (q2 > sensor_1_right_start) & (q2 < sensor_1_right_end)

sensor_2_left_start = 2.0 # um
sensor_2_left_end   = 2.5 # um

sensor_2_right_start = 2.0 # um
sensor_2_right_end   = 2.5 # um

sensor_2_left_indices  = (q2 > sensor_2_left_start ) & (q2 < sensor_2_left_end)
sensor_2_right_indices = (q2 > sensor_2_right_start) & (q2 < sensor_2_right_end)

tau_mr = np.arange(0.01, 0.091, 0.01)
tau_mr_1 = np.arange(0.1, 2.01, 0.1)
tau_mr = np.append(tau_mr, tau_mr_1)

resistance_array = []
resistance_nonlocal_array = []

io = PetscBinaryIO.PetscBinaryIO()

for index in tau_mr:
    print ('Index : ', index)

    filepath = 'dumps_wire_tau_%.2f'%index
    moment_files 		  = np.sort(glob.glob(filepath+'/moment*.h5'))
    lagrange_multiplier_files = \
            np.sort(glob.glob(filepath+'/lagrange_multipliers*.h5'))

    print (moment_files.size)
    
    dt = 0.025/4
    dump_interval = 5
    
    sensor_1_signal_array = []
    sensor_2_signal_array = []
    print("Reading sensor signal...")
    
    file_number = -1
    dump_file = moment_files[-1]
    
    moments = io.readBinaryFile(dump_file)
    moments = moments[0].reshape(N_q2, N_q1, 3)
    
    density = moments[:, :, 0]
    j_x     = moments[:, :, 1]
    j_y     = moments[:, :, 2]

    print ("Density : ", density.shape)
    
    left_edge = 0; right_edge = -1 

    sensor_1_left   = np.mean(density[sensor_1_left_indices, left_edge] )
    sensor_1_right  = np.mean(density[sensor_1_right_indices, right_edge])
    
    sensor_1_signal = sensor_1_left - sensor_1_right
    #current = np.mean(j_x[0, sensor_1_left_indices])

    resistance = sensor_1_signal#/current
    resistance_array.append(resistance)
    
    sensor_2_left   = np.mean(density[sensor_2_left_indices, left_edge] )
    sensor_2_right  = np.mean(density[sensor_2_right_indices, right_edge])
    sensor_2_signal = sensor_2_left - sensor_2_right
    
    resistance_nonlocal = sensor_2_signal#/current
    resistance_nonlocal_array.append(resistance_nonlocal)

    #pl.rcParams['figure.figsize']  = 12, 7.5
    

resistance_array = np.array(resistance_array)
resistance_nonlocal_array = np.array(resistance_nonlocal_array)
np.savetxt('tau_mr_wire.txt', tau_mr)
np.savetxt('resistance_vs_tau_mr_local_wire.txt', resistance_array)
np.savetxt('resistance_vs_tau_mr_nonlocal_wire.txt', resistance_nonlocal_array)

#params = [1000, 2, 10]
#popt, pcov = curve_fit(power_law, mu, resistance, p0=params)
#pl.subplot(211)
#pl.plot(tau_mr, resistance_array, '-o', label="local")
#pl.title("Local Resistance")
#pl.legend(loc="best")
#pl.plot(mu, power_law(mu, popt[0], popt[1], popt[2]), color = 'k', linestyle ='-')

#pl.xlabel('$\\tau_{mr} (ps)$')
#pl.ylabel('$\Omega$')

#pl.subplot(212)
#pl.title("Non-local Resistance")
#pl.xlabel('$\\tau_{mr} (ps)$')
#pl.ylabel('$\Omega$')
#pl.plot(tau_mr, resistance_nonlocal_array, '-o', label="non-local")
#pl.savefig('images/tau_mr_vs_R.png')
        

