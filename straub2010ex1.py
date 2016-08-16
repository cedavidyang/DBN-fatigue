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
from straub2010ex1_funcs import sys_prior, msr2m, msr2e

if __name__ == '__main__':
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
    r4 = Node("R4", parents=[r5], rvname='lognormal', rv=rv4)
    m4 = Node("M4", parents=[r4], rvname='continuous')
    m5 = Node("M5", parents=[r5], rvname='continuous')
    e = Node("E",parents=[r4,r5], rvname='discrete')

    # discretize continuous rv
    r4num = 20+1
    r5num = 20+1
    m4num = 20+2
    m5num = 20+2
    m = r5.rv.stats('m'); s = np.sqrt(r5.rv.stats('v'))
    lb = 50.; ub = 250.
    r4bins = np.hstack((0, np.linspace(lb, ub, r4num-1)))
    r4names = r4.discretize(lb, ub, r4num, infinity='+', bins=r4bins)
    m4names = m4.discretize(lb, ub, m4num, infinity='+-', bins=r4bins)
    lb = 50.; ub = 250.
    r5bins = np.hstack((0, np.linspace(lb, ub, r5num-1)))
    r5names = r5.discretize(lb, ub, r5num, infinity='+', bins=r5bins)
    m5names = m5.discretize(lb, ub, m5num, infinity='+-', bins=r5bins)

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
            probs.append(prob)
        probs = np.asarray(probs)
        if np.sum(probs) == 0:
            probs = np.ones(probs.shape)
        probs = probs/np.sum(probs)
        r4.assign_cpt(probs,label=np.asarray(label),statenames=r4names)

    # node M4
    nstate = m4num
    labels = itertools.product(np.arange(r4.nstates()))
    for i, label in enumerate(labels):
        truncrvs = []
        probs=[]
        for j, pstate in enumerate(label):
            truncrvs.append(m4.parents[j].truncate_rv(pstate))
        rvnames = ['u4', 'TE']
        rvs = [u4, rvte]
        trunclb = [rv.lb for rv in truncrvs]
        truncub = [rv.ub for rv in truncrvs]
        for k in xrange(nstate):
            mbins = [m4.bins[k], m4.bins[k+1]]
            rmean = float(truncrvs[0].stats('m'))
            prob = stats.norm.cdf(mbins[1], loc=rmean, scale=mstd) - \
                    stats.norm.cdf(mbins[0], loc=rmean, scale=mstd)
            prob0 = msr2m(rvnames, rvs, logmean, logstd, trunclb, truncub, mbins)
            probs.append(prob)
            print 'labels: {}, pf: {}, pf0:{}'.format((i,k), prob, prob0)
        probs = np.asarray(probs)
        if np.sum(probs) == 0:
            probs = np.ones(probs.shape)
        probs = probs/np.sum(probs,dtype=float)
        m4.assign_cpt(probs,label=np.asarray(label),statenames=m4names)

    # node M5
    nstate = m5num
    labels = itertools.product(np.arange(r5.nstates()))
    for i, label in enumerate(labels):
        truncrvs = []
        probs=[]
        for j, pstate in enumerate(label):
            truncrvs.append(m5.parents[j].truncate_rv(pstate))
        rvnames = ['u5', 'TE']
        rvs = [u5, rvte]
        trunclb = [rv.lb for rv in truncrvs]
        truncub = [rv.ub for rv in truncrvs]
        for k in xrange(nstate):
            mbins = [m5.bins[k], m5.bins[k+1]]
            prob = msr2m(rvnames, rvs, logmean, logstd, trunclb, truncub, mbins)
            probs.append(prob)
        probs = np.asarray(probs)
        if np.sum(probs) == 0:
            probs = np.ones(probs.shape)
        probs = probs/np.sum(probs,dtype=float)
        m5.assign_cpt(probs,label=np.asarray(label),statenames=m5names)

    # node E
    nstate = 2
    labels = itertools.product(np.arange(r4.nstates()),np.arange(r5.nstates()))
    labels = [label for label in labels]
    # labels = [(10,10),(4,4),(4,4)]
    for i,label in enumerate(labels):
        truncrvs=[]
        for j,pstate in enumerate(label):
            truncrvs.append(e.parents[j].truncate_rv(pstate))
        rvnames = ['ur', 'u1', 'u2', 'u3', 'u4', 'u5', 'h', 'v']
        rvs = [ur, u1, u2, u3, u4, u5, h, v]
        trunclb = [rv.lb for rv in truncrvs]
        truncub = [rv.ub for rv in truncrvs]
        syspf = msr2e(rvnames, rvs, logmean, logstd, rolnR, trunclb, truncub)
        probs = np.array([1.-syspf, syspf])
        e.assign_cpt(probs,label=np.asarray(label),statenames=['safe', 'fail'])
        print 'labels: {}, progress: {}%, pf: {}'.format(label, float(i)/len(labels)*100, syspf)

    # create new network
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


    # posterior belief 1
    m4measure = 50; m5measure=100
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
    r4beliefs = dbnet.get_node_beliefs(r4)
    r5beliefs = dbnet.get_node_beliefs(r5)
    r4rvs = [r4.truncate_rv(pstate) for pstate in np.arange(r4.nstates())]
    r5rvs = [r5.truncate_rv(pstate) for pstate in np.arange(r5.nstates())]
    np.savez('bndiscrete10.npz', r4beliefs=r4beliefs, r5beliefs=r5beliefs,
            r4bins=r4.bins, r5bins=r5.bins, r4rvs=r4rvs, r5rvs=r5rvs)

    # posterior belief 2
    dbnet.retract_nodefindings(m4)
    #dbnet.retract_netfindings()
    m4measure = 150; m5measure=100
    m4state = np.searchsorted(m4.bins, m4measure)-1
    if m4state<0: m4state=0
    dbnet.enter_finding(m4, m4state)
    #m5state = np.searchsorted(m5.bins, m5measure)-1
    #if m5state<0: m5state=0
    #dbnet.enter_finding(m5, m5state)
    beliefs = dbnet.get_node_beliefs(e)
    print "\nPosterior Belief:"
    for i in xrange(e.nstates()):
        print 'Given m4={:.1f} and m5={:.1f}, the probability of {} is {:f}'.format(m4measure, m5measure, e.statenames[i], beliefs[i])

    # posterior belief 3
    dbnet.retract_netfindings()
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
