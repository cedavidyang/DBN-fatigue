from pyNetica import Node, Network
from scipy import stats
import numpy as np
import itertools
import sys

from bnexample1_funcs import form2y

# random variables
rvX1 = stats.lognorm(1., scale=np.exp(0))
rvX3 = stats.lognorm(1., scale=np.exp(3*np.sqrt(2)))

# create nodes
x1 = Node("X1", parents=None, rvname='lognormal', rv=rvX1)
x2 = Node("X2", parents=[x1], rvname='continuous')
y = Node("Y", parents=[x2], rvname='discrete')

# discretize continuous rv
x1num = 5
x2num = 5
m = x1.rv.stats('m'); s = np.sqrt(x1.rv.stats('v'))
#lb = np.maximum(0, m-2*s); ub = m+2*s
lb = 0.; ub = m+1.5*s
x1names = x1.discretize(lb, ub, x1num, infinity='+')
x2names = x2.discretize(lb*10., ub*10., x2num, infinity='+')

# calculate and assign CPT
# node X1
x1cpt = x1.rv.cdf(x1.bins[1:]) - x1.rv.cdf(x1.bins[:-1])
x1cpt = x1cpt[np.newaxis,:]
x1.assign_cpt(x1cpt, statenames=x1names)
# node X2
lmd = 0.05
nstate = x2num
labels = itertools.product(np.arange(x1.nstates()))
for i, label in enumerate(labels):
    rvs = []
    probs=[]
    for j, pstate in enumerate(label):
        rvs.append(x2.parents[j].truncate_rv(pstate))
    for k in xrange(nstate):
        probs.append(rvs[0].cdf(x2.bins[k+1]/10.) -
                rvs[0].cdf(x2.bins[k]/10.))
    probs = np.asarray(probs)
    x2.assign_cpt(probs,label=np.asarray(label),statenames=x2names)
# node Y
nstate = 2
labels = itertools.product(np.arange(x2.nstates()))
rvnames = [x2.name, 'X3']
corr = np.eye(nstate)
for i, label in enumerate(labels):
    rvs = []
    probs=[]
    for j, pstate in enumerate(label):
        rvs.append(y.parents[j].truncate_rv(pstate, lmd=lmd))
    rvs.append(rvX3)
    pf = form2y(rvnames, rvs, corr)
    probs = np.array([1.-pf, pf])
    y.assign_cpt(probs,label=np.asarray(label),statenames=['neg', 'init'])

# create new network
dbnet = Network("BNexample1")

# add nodes to network
dbnet.add_nodes([x1, x2, y])
# add link: must before define nodes
dbnet.add_link()
# define nodes
dbnet.define_nodes()

# compile the net
dbnet.compile_net()
# enable autoupdate
dbnet.set_autoupdate()
# save the network
dbnet.save_net("BNexample1.dne")

# prior belief
beliefs = dbnet.get_node_beliefs(y)
print "Prior Belief:"
for i in xrange(y.nstates()):
    print 'The probability of {} is {:f}'.format(y.statenames[i], beliefs[i])

x1beliefs = dbnet.get_node_beliefs(x1)
statesvalues = x1.statenames.astype(float)
x1stats = x1.node_stats(x1beliefs)
print 'The mean of {} is {:f}'.format(x1.name, x1stats[0])
print 'The std of {} is {:f}'.format(x1.name, np.sqrt(x1stats[1]))

x2beliefs = dbnet.get_node_beliefs(x2)
statesvalues = x2.statenames.astype(float)
x2stats = x2.node_stats(x2beliefs, lmd=lmd)
print 'The mean of {} is {:f}'.format(x2.name, x2stats[0])
print 'The std of {} is {:f}'.format(x2.name, np.sqrt(x2stats[1]))
optx, dummy = x2.fit_stats('lognormal', x2beliefs, lmd=lmd)
print 'The mean of {} by fitting cdf is {:f}'.format(x2.name, optx[0])
print 'The std of {} by fitting cdf is {:f}'.format(x2.name, np.sqrt(optx[1]))

# posterior belief
dbnet.enter_finding(x1, 2)
beliefs = dbnet.get_node_beliefs(y)
print "\nPosterior Belief:"
for i in xrange(y.nstates()):
    print 'The probability of {} is {:f}'.format(y.statenames[i], beliefs[i])

#dbnet.enter_finding(y, 'init')
#x1beliefs = dbnet.get_node_beliefs(x1)
#statesvalues = x1.statenames.astype(float)
#x1stats = x1.node_stats(x1beliefs)
#print "\nPosterior Belief:"
#print 'The posterior mean of {} is {:f}'.format(x1.name, x1stats[0])
#print 'The posterior std of {} is {:f}'.format(x1.name, np.sqrt(x1stats[1]))
#optx, dummy = x1.fit_stats('lognormal', x1beliefs, lmd=lmd)
#print 'The mean of {} by fitting cdf is {:f}'.format(x1.name, optx[0])
#print 'The std of {} by fitting cdf is {:f}'.format(x1.name, np.sqrt(optx[1]))

#x2beliefs = dbnet.get_node_beliefs(x2)
#statesvalues = x2.statenames.astype(float)
#x2stats = x2.node_stats(x2beliefs, lmd=lmd)
#print 'The mean of {} is {:f}'.format(x2.name, x2stats[0])
#print 'The std of {} is {:f}'.format(x2.name, np.sqrt(x2stats[1]))
#optx, dummy = x2.fit_stats('lognormal', x2beliefs, lmd=lmd)
#print 'The mean of {} by fitting cdf is {:f}'.format(x2.name, optx[0])
#print 'The std of {} by fitting cdf is {:f}'.format(x2.name, np.sqrt(optx[1]))

# draw x1 pdf
x1max = 10
x1pts = np.linspace(0, x1max,100)
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':8})
plt.figure(figsize=(3.5,2.7))
plt.plot(x1pts, x1.node_pdf(x1pts, x1beliefs, lmd=lmd), '.')
plt.plot(x1pts, rvX1.pdf(x1pts))
plt.xlabel('X1'); plt.ylabel('PDF')
plt.subplots_adjust(left=0.15, bottom=0.15)
plt.legend(('BN', 'Theoretical'), fontsize=8)
#plt.savefig('x1priorpdf.png', dpi=1200)

# draw x1 cdf
x1max = 10
x1pts = np.linspace(0, x1max,100)
plt.figure(figsize=(3.5,2.7))
plt.plot(x1pts, x1.node_cdf(x1pts, x1beliefs, lmd=lmd), '.')
plt.plot(x1pts, rvX1.cdf(x1pts))
plt.xlabel('X1'); plt.ylabel('CDF')
plt.subplots_adjust(left=0.15, bottom=0.15)
plt.legend(('BN', 'Theoretical'), loc='lower right', fontsize=8)
#plt.savefig('x1priorcdf.png', dpi=1200)

# draw x2 pdf
x2max = 100
x2pts = np.linspace(0, x2max,100)
plt.figure(figsize=(3.5,2.7))
plt.plot(x2pts, x2.node_pdf(x2pts, x2beliefs, lmd=lmd), '.')
plt.plot(x2pts, rvX1.pdf(x2pts/10.))
plt.xlabel('X2'); plt.ylabel('PDF')
plt.subplots_adjust(left=0.15, bottom=0.15)
plt.legend(('BN', 'Theoretical'), fontsize=8)
plt.savefig('x2priorpdf.png', dpi=1200)

# draw x2 cdf
x2max = 100
x2pts = np.linspace(0, x2max,100)
plt.figure(figsize=(3.5,2.7))
plt.plot(x2pts, x2.node_cdf(x2pts, x2beliefs, lmd=lmd), '.')
plt.plot(x2pts, rvX1.cdf(x2pts/10.))
plt.xlabel('X2'); plt.ylabel('CDF')
plt.subplots_adjust(left=0.15, bottom=0.15)
plt.legend(('BN', 'Theoretical'), loc='lower right', fontsize=8)
plt.savefig('x2priorcdf.png', dpi=1200)
plt.close('all')

#=============================================================================#
#=============================================================================#
print "#=====================================================================#"

from scipy import stats
import numpy as np

# bnexample1.py should be implemented first

# simulation parameters
nsim = 1e6

# random variables
rvX1 = stats.lognorm(1., scale=np.exp(0))
rvX3 = stats.lognorm(1., scale=np.exp(3*np.sqrt(2)))

# start MC simulation for prior probs
x1smp = rvX1.rvs(size=nsim)
x3smp = rvX3.rvs(size=nsim)
x2smp= 10.*x1smp
ysmp = x3smp - x2smp
pf = np.sum(ysmp<0) / nsim
beliefs = [1-pf, pf]
ystatenames = ['neg', 'init']

print "Prior Belief with MC:"
for i in xrange(2):
    print 'The probability of {} is {:f}'.format(ystatenames[i], beliefs[i])
print 'The mean of X1 is {:f}'.format(np.mean(x1smp))
print 'The std of X1 is {:f}'.format(np.std(x1smp))
print 'The mean of X2 is {:f}'.format(np.mean(x2smp))
print 'The std of X2 is {:f}'.format(np.std(x2smp))

# start another MC for posterior probs given x1 in state 3
print "\nPosterior Belief with MC:"
ypostsmp = ysmp[np.logical_and(x1smp>x1.bins[2],
    x1smp<x1.bins[3])]
pf = np.sum(ypostsmp<0) / float(ypostsmp.size)
beliefs = [1-pf, pf]
for i in xrange(2):
    print 'The probability of {} is {:f}'.format(ystatenames[i], beliefs[i])
#print 'The Posterior mean of X1 is {:f}'.format(np.mean(x1smp[ysmp<0]))
#print 'The Posterior std of X1 is {:f}'.format(np.std(x1smp[ysmp<0]))
#print 'The Posterior mean of X2 is {:f}'.format(np.mean(x2smp[ysmp<0]))
#print 'The Posterior std of X2 is {:f}'.format(np.std(x2smp[ysmp<0]))

#import matplotlib.pyplot as plt
##plt.hist(x1smp, bins=200, normed=True)
##plt.hist(x1smp[ysmp<0], bins=200, normed=True)
##plt.xlim(0, 10)

#dtsorted = np.sort(x1smp[ysmp<0])
#yvals=np.arange(dtsorted.size)/float(dtsorted.size)
#plt.plot( dtsorted, yvals , label='MC')
#plt.xlim(0,20)
