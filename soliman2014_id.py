# soliman and frangopol (2014)

import numpy as np
from scipy import stats
import itertools

import os
import sys
import time
import datetime

from pyNetica import Node,Network
from pyNetica.netica import DECISION_NODE, UTILITY_NODE
from soliman2014_funcs import lognstats, wblstats
from soliman2014_funcs import ksmp_mc, aismp_mc, msr2k, mc2k, mc2ai

# parameters
trunclmd = 100.
acrit = 0.8
nsmp = int(1e6)
G = 1.12
lmd1 = -0.968; beta1 = -0.571    # w.r.t. mm
lmd2 =  0.122; beta2 = -0.305    # w.r.t. mm
lmd3 =  0.829; beta3 = -0.423    # w.r.t. mm
sigmae = 0.2    # mm
life=5; lifearray = np.arange(life)+1.
inspyear = (lifearray==4)
Cfail = 1000.
Cinit = 50.

def create_node_a(name, parents, ainum, ailb, aiub, aiedges, node_repair=None, asmp0=None):
    if node_repair is None:
        node_ai = Node(name, parents=parents, rvname='continuous')
    else:
        node_ai = Node(name, parents=parents+[node_repair], rvname='continuous')
    # dynamic discretization of nodes a and M
    node_ap = parents[0]
    knum = parents[1].nstates()
    mnum = parents[2].nstates()
    ainames = node_ai.discretize(ailb, aiub, ainum, infinity='+', bins=aiedges)
    aibins = node_ai.bins
    if node_repair is None:
        labels = itertools.product(np.arange(node_ap.nstates()), np.arange(knum),np.arange(mnum))
    else:
        labels = itertools.product(np.arange(node_ap.nstates()), np.arange(knum),
                np.arange(mnum), np.arange(2))
    labels = [label for label in labels]
    for i,label in enumerate(labels):
        if len(label)==4 and label[-1] == 1:
            binnum,dummy = np.histogram(asmp0, aibins)
            binnum[binnum==0] = 1e-12
            probs = binnum/np.sum(binnum, dtype=float)
            node_ai.assign_cpt(probs,label=np.asarray(label),statenames=node_ai.statenames)
        else:
            truncrvs=[]
            for j,pstate in enumerate(label):
                truncrvs.append(node_ai.parents[j].truncate_rv(pstate,lmd=trunclmd))
            rvnames = ['Ap', 'K', 'M']
            rvs = truncrvs[:3]
            probs,smpdb = mc2ai(rvnames, rvs, node_ai.bins, nsmp)
            # clean Ai states given Ai-1
            apstate = label[0]
            aplb = node_ap.bins[apstate]
            aiubs = node_ai.bins[1:]
            probs[aiubs<=aplb] = 0.
            probs = probs/np.sum(probs)
            node_ai.assign_cpt(probs,label=np.asarray(label),statenames=node_ai.statenames)
        # print 'labels: {}, progress: {}%, prob: {}'.format(label,
            # float(i)/len(labels)*100, np.array_str(probs,precision=3))
    return node_ai


def create_node_mi(name, parents, minum, milb, miub, mibins):
    node_mi = Node(name, parents=parents, rvname='continuous')
    minames = node_mi.discretize(milb, miub, minum, infinity='+-', bins=mibins)
    node_insp = parents[0]
    node_ai = parents[1]
    ainum = node_ai.nstates()
    labels = itertools.product(np.arange(node_insp.nstates()), np.arange(ainum))
    labels = [label for label in labels]
    for i,label in enumerate(labels):
        if label[0] == 0:    #no inspection
            probs = 1./minum * np.ones(minum)
        else:    # with insepction
            if label[0] == 1:
                lmd = lmd1; beta = beta1
            elif label[0] == 2:
                lmd = lmd2; beta = beta2
            elif label[0] == 3:
                lmd = lmd3; beta = beta3
            rvnames = ['Ai']
            rvs = [node_ai.truncate_rv(label[1], lmd=trunclmd)]
            aimean = rvs[0].stats('m')[()]
            rv_am = stats.norm(aimean, sigmae)
            pod = 1.-stats.norm.cdf((np.log(aimean)-lmd)/beta)
            probs = rv_am.cdf(node_mi.bins[1:])-rv_am.cdf(node_mi.bins[:-1])
            probs = probs/np.sum(probs)*pod
            probs[0] = probs[0]+(1.-pod)
        node_mi.assign_cpt(probs,label=np.asarray(label),statenames=node_mi.statenames)
        # print 'labels: {}, progress: {}%, prob: {}'.format(label,
            # float(i)/len(labels)*100, np.array_str(probs,precision=3))
    return node_mi


def set_node_repair(node_repair, acrit):
    node_insp = node_repair.parents[0]
    node_mi = node_repair.parents[1]
    nstate = node_repair.nstates()
    labels = itertools.product(np.arange(node_insp.nstates()),
            np.arange(node_mi.nstates()))
    labels = [label for label in labels]
    for i,label in enumerate(labels):
        truncrvs=[]
        probs=[]
        inspDecision = label[0]
        truncrv_mi = node_mi.truncate_rv(label[1],lmd=trunclmd)
        if inspDecision == 0:    # no inspection, i.e. no repair
            probs = np.array([1.0, 0.0])
            node_repair.assign_cpt(probs, label=np.asarray(label), statenames=node_repair.statenames)
        else:
            if truncrv_mi.lb>acrit:
                probs = np.array([0.0, 1.0])
            elif truncrv_mi.ub<acrit:
                probs = np.array([1.0, 0.0])
            else:
                prob = truncrv_mi.cdf(acrit)
                probs = np.array([prob, 1.-prob])
            node_repair.assign_cpt(probs, label=np.asarray(label), statenames=node_repair.statenames)
        # print 'labels: {}, progress: {}%, prob: {}'.format(label,
            # float(i)/len(labels)*100, np.array_str(probs,precision=3))


def inspection_utility(pstate):
    if pstate==0:
        return 0.
    else:
        return -(pstate+2.0)


if __name__ == '__main__':
    # random variables
    rv_a0 = stats.norm(0.5, 0.5*0.1)
    rv_m = stats.norm(3.0, 3.0*0.1)
    [logmean, logstd] = lognstats(2.3e-12, 0.3*2.3e-12)
    rv_C = stats.lognorm(logstd, scale=np.exp(logmean))
    [wblscale, wblc] = wblstats(22.5, 0.1*22.5)
    rv_Sre = stats.weibull_min(wblc, scale=wblscale)
    [logmean, logstd] = lognstats(5e6, 0.1*5e6)
    rv_Na = stats.lognorm(logstd, scale=np.exp(logmean))

    # network model
    # create time-independent nodes
    node_m = Node("M", parents=None, rvname='normal', rv=rv_m)
    node_k = Node("K", parents=[node_m], rvname='continuous')
    node_a0 = Node('a0', parents=None, rvname='normal', rv=rv_a0)
    # discretize continuous rv
    # node a0
    a0num = 5+2
    mu,var = rv_a0.stats(); sigma = np.sqrt(var)
    lb = mu-3.*sigma; ub = mu+3.*sigma
    a0bins = np.linspace(lb, ub, a0num-1)
    a0names = node_a0.discretize(lb, ub, a0num, infinity='+-', bins=a0bins)
    # node m
    mnum = 5+2
    mu,var = rv_m.stats(); sigma = np.sqrt(var)
    lb = mu-3.*sigma; ub = mu+3.*sigma
    mbins = np.linspace(lb, ub, mnum-1)
    mnames = node_m.discretize(lb, ub, mnum, infinity='+-', bins=mbins)
    # node k
    knum = 5+1
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
        probs = mc2k(rvnames, rvs, node_k.bins, G, nsmp)
        probs = probs/np.sum(probs,dtype=float)
        # print 'labels: {}, progress: {}%, pmf: {}'.format(label,
            # float(i)/len(labels)*100, np.array_str(probs,precision=3))
        node_k.assign_cpt(probs,label=np.asarray(label),statenames=node_k.statenames)


    # create time-dependent nodes
    aarray = [node_a0]
    marray = []    # measure array
    insparray = []    # inspection array
    rarray = []    # repair array
    uiarray = []    # utility of inspection
    urarray = []     # utility of repair
    aismp_prior = rv_a0.rvs(size=int(nsmp))
    for ia,rp in enumerate(inspyear):
        ainum = 5+1
        minum = 5+2
        aismp_prior = aismp_prior + 1e3*ksmp_prior*(aismp_prior*1e-3)**(msmp_prior/2.)
        ailb = np.percentile(aismp_prior, 0.1)
        aiub = np.percentile(aismp_prior, 99.9)
        if ailb>0:
            aiedges = np.hstack((0., np.linspace(ailb, aiub, ainum-1)))
        else:
            aiedges = np.linspace(0., aiub, ainum)
        if ia == 0:
            node_ai = create_node_a("A"+str(ia+1), [aarray[-1], node_k, node_m], ainum, ailb, aiub, aiedges)
            aarray.append(node_ai)
        elif inspyear[ia-1]:
            asmp0 = rv_a0.rvs(size=int(nsmp))
            node_ai = create_node_a("A"+str(ia+1), [aarray[-1], node_k, node_m], ainum, ailb, aiub, aiedges,
                    node_repair=rarray[-1], asmp0=asmp0)
            aarray.append(node_ai)
        elif inspyear[ia]:
            # node ai
            node_ai = create_node_a("A"+str(ia+1), [aarray[-1], node_k, node_m], ainum, ailb, aiub, aiedges)
            # node inspection
            node_insp = Node("Inspection"+str(ia+1), parents=None, rvname='discrete')
            node_insp.set_node_kind(DECISION_NODE)
            node_insp.set_node_state_name(['no', 'insp1', 'insp2', 'insp3'])
            # node mi
            node_mi = create_node_mi("M"+str(ia+1), [node_insp, node_ai], minum, ailb, aiub, aiedges)
            # node ui
            node_ui = Node("Util_Insp"+str(ia+1), parents=[node_insp], rvname='deterministic')
            node_ui.set_node_kind(UTILITY_NODE)
            node_ui.assign_func(inspection_utility)
            # node repair
            node_repair = Node("Repair"+str(ia+1), parents=[node_insp, node_mi], rvname='discrete')
            node_repair.set_node_kind(DECISION_NODE)
            node_repair.set_node_state_name(['no', 'repair'])
            set_node_repair(node_repair,acrit)
            # node ur
            node_ur = Node("Util_Repair"+str(ia+1), parents=[node_repair, node_ai], rvname='deterministic')
            node_ur.set_node_kind(UTILITY_NODE)
            def repair_utility(pstate):
                repairstate, aistate = pstate
                truncrv_ai = node_ai.truncate_rv(aistate, lmd=trunclmd)
                aimean = truncrv_ai.stats('m')[()]
                if repairstate == 0 and aimean<acrit:
                    utilr = 0.
                elif repairstate == 0 and aimean>acrit:
                    utilr = -Cfail
                elif repairstate == 1 and aimean<acrit:
                    rhoa = aimean/acrit
                    utilr = -(1.+1.-rhoa**0.5)*Cinit
                elif repairstate == 1 and aimean>acrit:
                    utilr = -Cinit
                return utilr
            node_ur.assign_func(repair_utility)
            aarray.append(node_ai)
            marray.append(node_mi)
            insparray.append(node_insp)
            rarray.append(node_repair)
            uiarray.append(node_ui)
            urarray.append(node_ur)
        else:
            node_ai = create_node_a("A"+str(ia+1), [aarray[-1], node_k, node_m], ainum, ailb, aiub, aiedges)
            aarray.append(node_ai)
        print 'year {}'.format(ia+1)


    # create new network
    dbnet = Network("Soliman2014InfDiag")

    # add nodes to network
    dbnet.add_nodes([node_m, node_k]+aarray+marray+insparray+uiarray+rarray+urarray)
    # add link: must be prior to defining nodes
    dbnet.add_link()
    # define nodes
    dbnet.define_nodes()

    # compile the net
    dbnet.compile_net()
    # enable autoupdate
    dbnet.set_autoupdate()
    # save the network
    dbnet.save_net("Soliman2014InfDiag.dne")


    #type 2
    dbnet.retract_netfindings()
    dummy = dbnet.get_node_expectedutils(node_insp)
    dummy = dbnet.get_node_expectedutils(node_repair)
    decision = dbnet.get_node_funcstate(node_repair, [2,4])
    print 'If the inspection decision is {} and meausre is {}, the best repair decision is {}.\n'.format(
            node_insp.statenames[2], marray[3][4], node_repair.statenames[decision])
