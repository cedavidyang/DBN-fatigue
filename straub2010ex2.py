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
from straub2010ex1_funcs import intg2h, msr2m, msr2q, msr2qcond, intg2e

if __name__ == '__main__':
    # random variables
    logmean = np.log(150./np.sqrt(1+0.2**2))
    logstd = np.sqrt(np.log(1+0.2**2))
    rolnR = 0.3
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

    # beta = (50.*0.4)/(np.pi/np.sqrt(6))
    # mu = 50.-np.euler_gamma*beta
    # h = stats.gumbel_r(loc=mu, scale=beta)
    urlogmean = np.log(35./np.sqrt(1+0.286**2))
    urlogstd = np.sqrt(np.log(1+0.286**2))
    uh = stats.lognorm(urlogstd, scale=np.exp(urlogmean))

    theta = (60*0.2)**2/60.
    k = 60./theta
    v = stats.gamma(k, scale=theta)


    # network model
    # create nodes
    r5 = Node("R5", parents=None, rvname='lognormal', rv=rv5)
    r4 = Node("R4", parents=[r5], rvname='lognormal', rv=rv4)
    m4 = Node("M4", parents=[r4], rvname='continuous')
    m5 = Node("M5", parents=[r5], rvname='continuous')
    q = Node("Q", parents=[r4,r5], rvname='continuous')
    uh = Node("Uh", parents=None, rvname='lognormal', rv=uh)
    e0 = Node("E0", parents=None, rvname='discrete')
    earray = [e0]
    life = 1; harray=[]
    for i in range(life):
        h = Node("H"+str(i+1), parents=[uh], rvname='continuous')
        e = Node("E"+str(i+1),parents=[earray[-1],q,h], rvname='discrete')
        harray.append(h)
        earray.append(e)

    # discretize continuous rv
    # r4, m4, r5 and m5
    r4num = 10+1
    r5num = 10+1
    m4num = 10+2
    m5num = 10+2
    m = r5.rv.stats('m'); s = np.sqrt(r5.rv.stats('v'))
    lb = 50.; ub = 250.
    r4bins = np.hstack((0, np.linspace(lb, ub, r4num-1)))
    r4names = r4.discretize(lb, ub, r4num, infinity='+', bins=r4bins)
    m4names = m4.discretize(lb, ub, m4num, infinity='+-', bins=r4bins)
    lb = 50.; ub = 250.
    r5bins = np.hstack((0, np.linspace(lb, ub, r5num-1)))
    r5names = r5.discretize(lb, ub, r5num, infinity='+', bins=r5bins)
    m5names = m5.discretize(lb, ub, m5num, infinity='+-', bins=r5bins)
    # q
    qnum = 10+1
    qlb = 0.; qub = 150.
    qbins = np.hstack(np.linspace(qlb, qub, qnum))
    qnames = q.discretize(qlb, qub, qnum, infinity='+', bins=qbins)
    # uh
    uhnum = 10+1
    uhlb = 0.; uhub = 150.
    uhbins = np.hstack(np.linspace(uhlb, uhub, uhnum))
    uhnames = uh.discretize(uhlb, uhub, uhnum, infinity='+', bins=uhbins)
    # h
    for h in harray:
        hnum = 10+1
        hlb = 0.; hub = 150.
        hbins = np.hstack(np.linspace(hlb, hub, hnum))
        hnames = h.discretize(hlb, hub, hnum, infinity='+', bins=hbins)


    # calculate and assignCPT
    # e0
    probs = np.array([[1., 0.]])
    e0.assign_cpt(probs, statenames=['safe', 'fail'])
    # node R5
    r5cpt = r5.rv.cdf(r5.bins[1:]) - r5.rv.cdf(r5.bins[:-1])
    r5cpt = r5cpt[np.newaxis,:]
    r5.assign_cpt(r5cpt, statenames=r5names)
    # node Uh
    uhcpt = uh.rv.cdf(uh.bins[1:]) - uh.rv.cdf(uh.bins[:-1])
    uhcpt = uhcpt[np.newaxis,:]
    uh.assign_cpt(uhcpt, statenames=uhnames)
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
    # Harray
    for h in harray:
        nstate = hnum
        labels = itertools.product(np.arange(uh.nstates()))
        for i, label in enumerate(labels):
            truncrvs = []
            probs=[]
            for j, pstate in enumerate(label):
                truncrvs.append(h.parents[j].truncate_rv(pstate))
            trunclb = [rv.lb for rv in truncrvs]
            truncub = [rv.ub for rv in truncrvs]
            for k in xrange(nstate):
                hbin = [h.bins[k], h.bins[k+1]]
                prob = intg2h(truncrvs[0], trunclb, truncub, hbin)
                print 'labels: {}, prob:{}'.format((i,k), prob)
                probs.append(prob)
            probs = np.asarray(probs)
            probs = probs/np.sum(probs,dtype=float)
            h.assign_cpt(probs,label=np.asarray(label),statenames=h.statenames)
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
            prob0 = stats.norm.cdf(mbins[1], loc=rmean, scale=mstd) - \
                    stats.norm.cdf(mbins[0], loc=rmean, scale=mstd)
            prob = msr2m(rvnames, rvs, logmean, logstd, trunclb, truncub, mbins)
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
    # node Q
    nstate = qnum
    labels = itertools.product(np.arange(r4.nstates()),np.arange(r5.nstates()))
    labels = [label for label in labels]
    for i,label in enumerate(labels):
        truncrvs=[]
        probs=[]
        for j,pstate in enumerate(label):
            truncrvs.append(q.parents[j].truncate_rv(pstate))
        rvnames = ['ur', 'u1', 'u2', 'u3', 'u4', 'u5', 'v']
        rvs = [ur, u1, u2, u3, u4, u5, v]
        trunclb = [rv.lb for rv in truncrvs]
        truncub = [rv.ub for rv in truncrvs]
        pcond = msr2qcond(rvnames, rvs, logmean, logstd, rolnR, trunclb, truncub)
        for k in xrange(nstate-1):
            qbin = [q.bins[k], q.bins[k+1]]
            prob = msr2q(rvnames, rvs, logmean, logstd, rolnR, trunclb, truncub, qbin)
            if len(probs) and prob>probs[-1]: prob = probs[-1]
            print 'labels: {}, progress: {}%, cdf: {}'.format(np.hstack((label,k)),
                    float(i)/len(labels)*100, prob)
            probs.append(prob)
        probs = np.hstack((pcond, probs, 0.))/pcond
        probs[probs>1.] = 1.
        probs = probs[:-1]-probs[1:]
        q.assign_cpt(probs,label=np.asarray(label),statenames=q.statenames)
    # node E
    nstate = qnum
    for e,h in zip(earray[1:],harray):
        labels = itertools.product(np.arange(2), np.arange(q.nstates()),np.arange(h.nstates()))
        labels = [label for label in labels]
        i = 0
        for i,label in enumerate(labels):
            truncrvs=[]
            for j,pstate in enumerate(label[1:]):
                truncrvs.append(e.parents[j+1].truncate_rv(pstate))
            if label[0] == 1:    # failure in previous time slot
                prob = 1.
            else:
                prob = intg2e(truncrvs[0], truncrvs[1])
            print 'labels: {}, progress: {}%, prob: {}'.format(label,
                    float(i)/len(labels)*100, prob)
            probs = np.array([1.-prob, prob])
            e.assign_cpt(probs,label=np.asarray(label),statenames=['safe','fail'], labels=labels)

    # create new network
    dbnet = Network("Straub2010Ex2")

    # add nodes to network
    dbnet.add_nodes([r5, r4, m5, m4, uh, q]+harray+earray)
    # add link: must be prior to defining nodes
    dbnet.add_link()
    # define nodes
    dbnet.define_nodes()

    # compile the net
    dbnet.compile_net()
    # enable autoupdate
    dbnet.set_autoupdate()
    # save the network
    dbnet.save_net("Straub2010Ex2.dne")

    # # prior belief
    # beliefs = dbnet.get_node_beliefs(e)
    # print "Prior Belief:"
    # for i in xrange(e.nstates()):
        # print 'The probability of {} is {:f}'.format(e.statenames[i], beliefs[i])

    # # posterior belief 1
    # m4measure = 50; m5measure=100
    # m4state = np.searchsorted(m4.bins, m4measure)-1
    # if m4state<0: m4state=0
    # m5state = np.searchsorted(m5.bins, m5measure)-1
    # if m5state<0: m5state=0
    # dbnet.enter_finding(m4, m4state)
    # dbnet.enter_finding(m5, m5state)
    # beliefs = dbnet.get_node_beliefs(e)
    # print "\nPosterior Belief:"
    # for i in xrange(e.nstates()):
        # print 'Given m4={:.1f} and m5={:.1f}, the probability of {} is {:f}'.format(m4measure, m5measure, e.statenames[i], beliefs[i])
