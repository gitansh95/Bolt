import pylab as pl
import numpy as np



mask = np.loadtxt("fields/E_x_1.txt")


pl.contourf(mask, 100, cmap = 'bwr')
pl.gca().set_aspect(1)

print(mask.shape)

pl.savefig('images/iv.png')
pl.clf()
