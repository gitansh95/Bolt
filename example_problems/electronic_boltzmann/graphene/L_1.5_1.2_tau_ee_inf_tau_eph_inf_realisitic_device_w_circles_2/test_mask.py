import pylab as pl
import numpy as np



mask = np.loadtxt("fields/E_x_1.txt")
print("E_x shape : ", mask.shape)

mask = np.loadtxt("mask.txt")

print("shape mask : ", mask.shape)

pl.contourf(mask.T, 100, cmap = 'bwr')
pl.gca().set_aspect(1)
pl.colorbar()

pl.savefig("images/iv2.png")
pl.clf()



# Rank 0
N_q1_local = 70
N_q2_local = 82
E_x = mask[:int(N_q1_local), :int(N_q2_local)]

pl.subplot(2, 3, 1)
pl.contourf(E_x.T, 100, cmap = 'gray')
pl.gca().set_aspect(1)
pl.title("Rank 0")

# Rank 1
N_q1_local = 69
N_q2_local = 82
E_x = mask[68:68+int(N_q1_local), :int(N_q2_local)]

pl.subplot(2, 3, 2)
pl.contourf(E_x.T, 100, cmap = 'gray')
pl.gca().set_aspect(1)
pl.title("Rank 1")

# Rank 2
E_x = mask[-int(N_q1_local):, :int(N_q2_local)]

pl.subplot(2, 3, 3)
pl.contourf(E_x.T, 100, cmap = 'gray')
pl.gca().set_aspect(1)
pl.title("Rank 2")

# Rank 3
N_q1_local = 70
N_q2_local = 82
E_x = mask[:int(N_q1_local), -int(N_q2_local):]

pl.subplot(2, 3, 4)
pl.contourf(E_x.T, 100, cmap = 'gray')
pl.gca().set_aspect(1)
pl.title("Rank 3")

        
# Rank 4
N_q1_local = 69
N_q2_local = 82
E_x = mask[68:68+int(N_q1_local), -int(N_q2_local):]

pl.subplot(2, 3, 5)
pl.contourf(E_x.T, 100, cmap = 'gray')
pl.gca().set_aspect(1)
pl.title("Rank 4")

# Rank 5 
E_x = mask[-int(N_q1_local):, -int(N_q2_local):]

pl.subplot(2, 3, 6)
pl.contourf(E_x.T, 100, cmap = 'gray')
pl.gca().set_aspect(1)
pl.title("Rank 5")

pl.savefig('images/iv.png')
pl.clf()
