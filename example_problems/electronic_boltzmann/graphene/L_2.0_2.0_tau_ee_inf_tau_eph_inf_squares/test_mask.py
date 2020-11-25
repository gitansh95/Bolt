import pylab as pl
import numpy as np



#mask = np.loadtxt("mask.txt")
mask = np.loadtxt("fields/E_y_0.txt")


pl.contourf(mask, 100, cmap = 'bwr')
pl.gca().set_aspect(1)

print(mask.shape)

pl.savefig('images/iv.png')
pl.clf()
