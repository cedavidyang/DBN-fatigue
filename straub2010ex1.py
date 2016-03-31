# example 1 (the simple frame model) in Straub and Der Kiureghian (2010;
# application)

import numpy as np
from scipy import stats
import itertools

import os
import sys
import time
import datetime

from pyNetica import Node,Network
from straub2010ex1_funcs import sys_prior, form2m, msr2e

# random variables
logmean = np.log(150./np.sqrt(1+0.2**2))
logstd = np.sqrt(np.log(1+0.2**2))
rv4 = stats.lognorm(logstd, scale=np.exp(logmean))
rv5 = stats.lognorm(logstd, scale=np.exp(logmean))
ur = stats.norm()
u1 = stats.norm()
u2 = stats.norm()
u3 = stats.norm()
u4 = stats.norm()
u5 = stats.norm()
mmean=0.; mstd=15.
rvte = stats.norm(loc=mmean, scale=mstd)    # testing error, normal with mean=0, std=15kNm

beta = (50.*0.4)/(np.pi/np.sqrt(6))
mu = 50.-np.euler_gamma*beta
h = stats.gumbel_r(loc=mu, scale=beta)

theta = (60*0.2)**2/60.
k = 60./theta
v = stats.gamma(k, scale=theta)

rolnR = 0.3

# prior reliability using pyStRe
rvnames = ['ur', 'u1', 'u2', 'u3', 'u4', 'u5', 'h', 'v']
rvs = [ur, u1, u2, u3, u4, u5, h, v]
syspf = sys_prior(rvnames, rvs, rolnR)
print "Prior system failure probability is {}".format(syspf)
print "Prior system reliability is {}".format(stats.norm.ppf(1-syspf))

# network model
# create nodes
r5 = Node("R5", parents=None, rvname='lognormal', rv=rv5)
r4 = Node("R4", parents=[r5], rvname='continuous')
m4 = Node("M4", parents=[r4], rvname='continuous')
m5 = Node("M5", parents=[r5], rvname='continuous')
e = Node("E",parents=[r4,r5], rvname='discrete')

# discretize continuous rv
r4num = 5
r5num = 5
m4num = 6
m5num = 6
m = r5.rv.stats('m'); s = np.sqrt(r5.rv.stats('v'))
lb = 0.; ub = m+1.5*s
r5names = r5.discretize(lb, ub, r5num, infinity='+')
r4names = r4.discretize(lb, ub, r4num, infinity='+')
m4names = m4.discretize(lb, ub, m4num, infinity='+-')
m5names = m5.discretize(lb, ub, m4num, infinity='+-')

# calculate and assignCPT
# node R5
r5cpt = r5.rv.cdf(r5.bins[1:]) - r5.rv.cdf(r5.bins[:-1])
r5cpt = r5cpt[np.newaxis,:]
r5.assign_cpt(r5cpt, statenames=r5names)
# node R4
nstate = r4num
labels = itertools.product(np.arange(r5.nstates()))
mvnorm = stats.multivariate_normal(mean=None,cov=np.array([[1.,rolnR],[rolnR,1.]]))
for i,label in enumerate(labels):
    probs=[]
    plb = r5.bins[label[0]]    #r5 lower bound
    pub = r5.bins[label[0]+1]    #r5 upper bound
    z5lb = (np.log(plb)-logmean)/logstd
    z5ub = (np.log(pub)-logmean)/logstd
    for k in xrange(nstate):
        clb = r4.bins[k]    #r4 lower bound
        cub = r4.bins[k+1]    #r4 upper bound
        z4lb = (np.log(clb)-logmean)/logstd
        z4ub = (np.log(cub)-logmean)/logstd
        low = np.array([z5lb, z4lb]); low[np.isneginf(low)] = -20.
        upp = np.array([z5ub, z4ub]); upp[np.isposinf(upp)] = 20.
        prob,info = stats.mvn.mvnun(low, upp, np.zeros(2), np.array([[1.,rolnR],[rolnR,1.]]))
        #prob = prob/(stats.norm.cdf(z5ub)-stats.norm.cdf(z5lb))
        probs.append(prob)
    probs = np.asarray(probs)
    probs = probs/np.sum(probs)
    r4.assign_cpt(probs,label=np.asarray(label),statenames=r4names)
# node M4
nstate = m4num
labels = itertools.product(np.arange(r4.nstates()))
rvnames = [r4.name, 'TE']
for i, label in enumerate(labels):
    rvs = []
    probs=[]
    for j, pstate in enumerate(label):
        rvs.append(m4.parents[j].truncate_rv(pstate))
    rvs.append(rvte)
    corr = np.eye(len(rvs))
    for k in xrange(nstate):
        bins = [m4.bins[k], m4.bins[k+1]]
        ## get prob from FORM
        #prob0 = form2m(rvnames, rvs, corr, bins)
        # get approximate prob on the condition of the mean value of Ri
        rmean = float(rvs[0].stats('m'))
        prob = stats.norm.cdf(bins[1], loc=rmean, scale=mstd) - \
                stats.norm.cdf(bins[0], loc=rmean, scale=mstd)
        probs.append(prob)
    probs = np.asarray(probs)
    probs = probs/np.sum(probs)
    m4.assign_cpt(probs,label=np.asarray(label),statenames=m4names)
# node M5
nstate = m5num
labels = itertools.product(np.arange(r5.nstates()))
rvnames = [r5.name, 'TE']
for i, label in enumerate(labels):
    rvs = []
    probs=[]
    for j, pstate in enumerate(label):
        rvs.append(m5.parents[j].truncate_rv(pstate))
    rvs.append(rvte)
    corr = np.eye(len(rvs))
    for k in xrange(nstate):
        bins = [m5.bins[k], m5.bins[k+1]]
        ## get prob from FORM
        #prob0 = form2m(rvnames, rvs, corr, bins)
        # get approximate prob on the condition of the mean value of Ri
        rmean = float(rvs[0].stats('m'))
        prob = stats.norm.cdf(bins[1], loc=rmean, scale=mstd) - \
                stats.norm.cdf(bins[0], loc=rmean, scale=mstd)
        probs.append(prob)
    probs = np.asarray(probs)
    probs = probs/np.sum(probs)
    m5.assign_cpt(probs,label=np.asarray(label),statenames=m5names)
# node E
nstate = 2
labels = itertools.product(np.arange(r4.nstates()),np.arange(r5.nstates()))
labels = [label for label in labels]
allpfs = []
for i,label in enumerate(labels):
    print 'labels: {}, progress: {}\%'.format(label, float(i)/len(labels)*100)
    rvs=[]
    for j,pstate in enumerate(label):
        rvs.append(e.parents[j].truncate_rv(pstate))
    rvnames = ['ur', 'u1', 'u2', 'u3', 'r4', 'r5', 'h', 'v']
    rvs = [ur, u1, u2, u3]+rvs+[h, v]
    syspf = msr2e(rvnames, rvs, logmean, logstd, rolnR)
    probs = np.array([1.-syspf, syspf])
    allpfs.append(syspf)
    e.assign_cpt(probs,label=np.asarray(label),statenames=['safe', 'fail'])

# create new network
import ipdb; ipdb.set_trace() # BREAKPOINT
dbnet = Network("Straub2010Ex1")

# add nodes to network
dbnet.add_nodes([r5, r4, m5, m4, e])
# add link: must be prior to defining nodes
dbnet.add_link()
# define nodes
dbnet.define_nodes()

# compile the net
dbnet.compile_net()
# enable autoupdate
dbnet.set_autoupdate()
# save the network
dbnet.save_net("Straub2010Ex1.dne")

# prior belief
beliefs = dbnet.get_node_beliefs(e)
print "Prior Belief:"
for i in xrange(e.nstates()):
    print 'The probability of {} is {:f}'.format(e.statenames[i], beliefs[i])

# posterior belief
m4measure = 150; m5measure=200
m4state = np.searchsorted(m4.bins, m4measure)-1
if m4state<0: m4state=0
m5state = np.searchsorted(m5.bins, m5measure)-1
if m5state<0: m5state=0
dbnet.enter_finding(m4, m4state)
dbnet.enter_finding(m5, m5state)
beliefs = dbnet.get_node_beliefs(e)
print "\nPosterior Belief:"
for i in xrange(e.nstates()):
    print 'Given m4={:.1f} and m5={:.1f}, the probability of {} is {:f}'.format(m4measure, m5measure, e.statenames[i], beliefs[i])
