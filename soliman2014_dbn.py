# soliman and frangopol (2014)

import numpy as np
from scipy import stats
import itertools

import os
import sys
import time
import datetime

from pyNetica import Node,Network
from soliman2014_funcs import lognstats, wblstats
from soliman2014_funcs import ksmp_mc, aismp_mc, msr2k, mc2k, mc2ai

trunclmd = 100.

if __name__ == '__main__':
    # parameters
    nsmp = 1e6
    G = 1.12
    lmd = 0.122; beta = -0.305    # w.r.t. mm
    sigmae = 0.2    # mm
    acrit = 5.
    life=5; lifearray = np.arange(life)+1.
    # random variables
    rv_a0 = stats.norm(0.5, 0.5*0.1)
    rv_m = stats.norm(3.0, 3.0*0.1)
    # [logmean, logstd] = lognstats(2.3e-12, 0.3*2.3e-12)
    [logmean, logstd] = lognstats(4.5e-13, 0.3*4.5e-13)
    rv_C = stats.lognorm(logstd, scale=np.exp(logmean))
    [wblscale, wblc] = wblstats(22.5, 0.1*22.5)
    rv_Sre = stats.weibull_min(wblc, scale=wblscale)
    [logmean, logstd] = lognstats(2e6, 0.1*2e6)
    rv_Na = stats.lognorm(logstd, scale=np.exp(logmean))

    # network model
    # create nodes
    node_m = Node("M", parents=None, rvname='normal', rv=rv_m)
    node_k = Node("K", parents=[node_m], rvname='continuous')
    node_a0 = Node('a0', parents=None, rvname='normal', rv=rv_a0)
    aarray = [node_a0]
    marray=[]
    for i in range(life):
        ai = Node("A"+str(i+1), parents=[aarray[-1], node_k, node_m], rvname='continuous')
        mi = Node("M"+str(i+1), parents=[ai], rvname='continuous')
        aarray.append(ai)
        marray.append(mi)

    # discretize continuous rv
    # node a0
    a0num = 10+2
    mu,var = rv_a0.stats(); sigma = np.sqrt(var)
    lb = mu-3.*sigma; ub = mu+3.*sigma
    a0bins = np.linspace(lb, ub, a0num-1)
    a0names = node_a0.discretize(lb, ub, a0num, infinity='+-', bins=a0bins)
    # node m
    mnum = 10+2
    mu,var = rv_m.stats(); sigma = np.sqrt(var)
    lb = mu-3.*sigma; ub = mu+3.*sigma
    mbins = np.linspace(lb, ub, mnum-1)
    mnames = node_m.discretize(lb, ub, mnum, infinity='+-', bins=mbins)
    # node k
    knum = 10+1
    ksmp_prior,msmp_prior = ksmp_mc(nsmp, rv_C, rv_Sre, G, rv_m, rv_Na)
    klb = np.percentile(ksmp_prior, 5)
    kub = np.percentile(ksmp_prior, 95)
    if klb>0:
        kbins = np.hstack((0., np.linspace(klb, kub, knum-1)))
    else:
        kbins = np.linspace(0, kub, knum)
    knames = node_k.discretize(klb, kub, knum, infinity='+', bins=kbins)

    # calculate and assign CPT
    # node a0
    a0cpt = node_a0.rv.cdf(node_a0.bins[1:]) - node_a0.rv.cdf(node_a0.bins[:-1])
    a0cpt = a0cpt[np.newaxis, :]
    node_a0.assign_cpt(a0cpt, statenames=a0names)
    # node m
    mcpt = node_m.rv.cdf(node_m.bins[1:]) - node_m.rv.cdf(node_m.bins[:-1])
    mcpt = mcpt[np.newaxis, :]
    node_m.assign_cpt(mcpt, statenames=mnames)
    # node k
    nstate = knum
    labels = itertools.product(np.arange(node_m.nstates()))
    labels = [label for label in labels]
    for i,label in enumerate(labels):
        truncrvs = []
        for j, pstate in enumerate(label):
            truncrvs.append(node_k.parents[j].truncate_rv(pstate,lmd=trunclmd))
        rvnames = ['M', 'C', 'Sre', 'Na']
        rvs = truncrvs+[rv_C, rv_Sre, rv_Na]
        # rvnames = ['M', 'C', 'Sre', 'Na']
        # rvs = [rv_m, rv_C, rv_Sre, rv_Na]
        probs = mc2k(rvnames, rvs, node_k.bins, G, nsmp)
        probs = probs/np.sum(probs,dtype=float)
        print 'labels: {}, progress: {}%, pmf: {}'.format(label,
            float(i)/len(labels)*100, np.array_str(probs,precision=3))
        node_k.assign_cpt(probs,label=np.asarray(label),statenames=node_k.statenames)
    # node ai
    aismp_prior = rv_a0.rvs(size=int(nsmp))
    for ia, (node_ai,node_mi) in enumerate(zip(aarray[1:], marray)):
        # dynamic discretization of nodes a and M
        node_ap = aarray[ia]
        ainum = 10+1
        minum = 10+2
        aismp_prior = aismp_mc(nsmp, 1., aismp_prior, rv_C, rv_Sre, G, rv_m, rv_Na, acrit)
        ailb = np.percentile(aismp_prior[aismp_prior<acrit], 0.1)
        aiub = np.percentile(aismp_prior[aismp_prior<acrit], 99.9)
        if ailb>0:
            aibins = np.hstack((0., np.linspace(ailb, aiub, ainum-1)))
        else:
            aibins = np.linspace(0., aiub, ainum)
        ainames = node_ai.discretize(ailb, aiub, ainum, infinity='+', bins=aibins)
        minames = node_mi.discretize(ailb, aiub, minum, infinity='+-', bins=aibins)
        nstate = ainum
        # ai = Node("A"+str(i+1), parents=[aarray[-1], node_k, node_m], rvname='continuous')
        labels = itertools.product(np.arange(node_ap.nstates()), np.arange(knum),np.arange(mnum))
        labels = [label for label in labels]
        for i,label in enumerate(labels):
            truncrvs=[]
            for j,pstate in enumerate(label):
                truncrvs.append(node_ai.parents[j].truncate_rv(pstate,lmd=trunclmd))
            rvnames = ['Ap', 'K', 'M']
            rvs = truncrvs
            probs,smpdb = mc2ai(rvnames, rvs, node_ai.bins, acrit, nsmp=nsmp)
            # clean Ai states given Ai-1
            apstate = label[0]
            aplb = node_ap.bins[apstate]
            aiubs = node_ai.bins[1:]
            probs[aiubs<=aplb] = 0.
            probs = probs/np.sum(probs)
            print 'labels: {}, progress: {}%, prob: {}'.format(label,
                float(i)/len(labels)*100, np.array_str(probs,precision=3))
            node_ai.assign_cpt(probs,label=np.asarray(label),statenames=node_ai.statenames)
        # node mi
        nstate = minum
        labels = itertools.product(np.arange(ainum))
        labels = [label for label in labels]
        for i,label in enumerate(labels):
            truncrvs=[]
            probs=[]
            for j,pstate in enumerate(label):
                truncrvs.append(node_mi.parents[j].truncate_rv(pstate,lmd=trunclmd))
            rvnames = ['Ai']
            rvs = truncrvs
            aimean = rvs[0].stats('m')[()]
            rv_am = stats.norm(aimean, sigmae)
            pod = 1.-stats.norm.cdf((np.log(aimean)-lmd)/beta)
            probs = rv_am.cdf(node_mi.bins[1:])-rv_am.cdf(node_mi.bins[:-1])
            print 'labels: {}, progress: {}%, prob: {}'.format(label,
                float(i)/len(labels)*100, np.array_str(probs,precision=3))
            probs = probs/np.sum(probs)*pod
            probs[0] = probs[0]+(1.-pod)
            node_mi.assign_cpt(probs,label=np.asarray(label),statenames=node_mi.statenames)

    # create new network
    dbnet = Network("Soliman2014DBN")

    # add nodes to network
    dbnet.add_nodes([node_m, node_k]+aarray+marray)
    # add link: must be prior to defining nodes
    dbnet.add_link()
    # define nodes
    dbnet.define_nodes()

    # compile the net
    dbnet.compile_net()
    # enable autoupdate
    dbnet.set_autoupdate()
    # save the network
    dbnet.save_net("Soliman2014DBN.dne")

    # inferences
    pfarray = []
    for ai in aarray:
        beliefs = dbnet.get_node_beliefs(ai)
        aistate = np.searchsorted(ai.bins, acrit)-1
        aitruncrv = ai.truncate_rv(aistate,lmd=trunclmd)
        lt = np.sum(beliefs[:aistate])+beliefs[aistate]*aitruncrv.cdf(acrit)
        pfarray.append(1-lt)

    aimeanarray = []
    aistdarray = []
    for ai in aarray:
        beliefs = dbnet.get_node_beliefs(ai)
        aimean,aivar = ai.node_stats(beliefs,lmd=trunclmd)
        aimeanarray.append(aimean)
        aistdarray.append(np.sqrt(aivar))
