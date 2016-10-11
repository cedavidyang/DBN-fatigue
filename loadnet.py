import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import sys

from pyNetica import Network

def statename2edges(statenames):
    edges = []
    for statename in statenames:
        numstr = statename[1:].replace('_', '.')
        if numstr != 'inf':
            edges.append(numstr)
        else:
            continue
    return np.array(edges, dtype=float)


if __name__ == '__main__':
    # load net from dbn file
    dbnet = Network('Soliman2014DBN', file='Soliman2014DBN.dne')
    # compile the net
    dbnet.compile_net()
    # enable autoupdate
    dbnet.set_autoupdate()

    m3 = dbnet.find_nodenamed('M3')
    if m3 == 'error':
        print "ERROR: cannot find node M3"
        sys.exit(1)
    m3meaure = 0.6
    m3edges = statename2edges(m3.statenames)
    m3bins = np.hstack((-np.inf, m3edges, np.inf))
    m3state = np.searchsorted(m3bins, m3meaure)-1
    dbnet.enter_finding(m3, m3state)
    a5 = dbnet.find_nodenamed('A5')
    if a5 == 'error':
        print "ERROR: cannot find node A5"
        sys.exit(1)
    a5edges = statename2edges(a5.statenames)
    a5bins = np.hstack((0., a5edges, np.inf))
    a5.set_bins(a5bins)
    beliefs = dbnet.get_node_beliefs(a5)
    dbncdf = a5.node_cdf(a5.bins[1:-1], beliefs, lmd=100.)

    # comparison
    asmpmcmc = np.load('asmpmcmc.npy')
    a = asmpmcmc[-1,:]
    num_bins = 20
    counts, bin_edges = np.histogram(a, bins=num_bins, normed=True)
    cdf = np.cumsum(counts, dtype=float)/np.sum(counts)
    
    # plot
    plt.plot(a5.bins[1:-1], dbncdf)
    plt.plot(bin_edges[1:], cdf)
