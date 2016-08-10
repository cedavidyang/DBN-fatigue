import dill
import pymc
# from pyNetica import Node,Network
import numpy as np
import matplotlib.pyplot as plt

# load data
# with np.load('bndiscrete20std0.npz') as data:
with np.load('bndiscrete10.npz') as data:
    r4beliefs = data['r4beliefs']
    r5beliefs = data['r5beliefs']
    r4bins = data['r4bins']
    r5bins = data['r5bins']
    # r4rep = data['r4means']
    # r5rep = data['r5means']
    r4rvs = data['r4rvs']
    r5rvs = data['r5rvs']
db=pymc.database.pickle.load('pymc2_straub2010ex1_m45_50_100.pickle')
r4smp = db.trace('r4')[:]
r5smp = db.trace('r5')[:]
db.close()

# test
r4rep = [rv.stats('m')[()] for rv in r4rvs]
r5rep = [rv.stats('m')[()] for rv in r5rvs]
r4mean = np.dot(r4rep, r4beliefs)
r5mean = np.dot(r5rep, r5beliefs)

# plot
# posterior r4
# netica
plt.plot(r4bins[1:-1], np.cumsum(r4beliefs)[:-1], 'o')
# pymc
r4smpsorted = np.sort(r4smp)
# calculate the proportional values of samples
p = 1. * np.arange(r4smp.size) / (r4smp.size - 1)
plt.plot(r4smpsorted, p)
# posterior r5
# netica
plt.figure()
plt.plot(r5bins[1:-1], np.cumsum(r5beliefs)[:-1], 'o')
# pymc
r5smpsorted = np.sort(r5smp)
# calculate the proportional values of samples
p = 1. * np.arange(r5smp.size) / (r5smp.size - 1)
plt.plot(r5smpsorted, p)
