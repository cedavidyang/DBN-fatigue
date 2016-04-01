import os
strepath = os.path.join(os.path.expanduser('~'), 'FERUM')
import sys
sys.path.append(strepath)

from pyStRe import *

import numpy as np
from scipy import stats

# straub2010ex1: prior reliability
def sys_prior(rvnames, rvs, rolnR):
    logmean = np.log(150./np.sqrt(1+0.2**2))
    logstd = np.sqrt(np.log(1+0.2**2))
    ## limit state 1 (use correlated rv)
    #logr1 = stats.norm(loc=logmean, scale=logstd)
    #logr2 = stats.norm(loc=logmean, scale=logstd)
    #logr3 = stats.norm(loc=logmean, scale=logstd)
    #logr4 = stats.norm(loc=logmean, scale=logstd)
    #logr5 = stats.norm(loc=logmean, scale=logstd)
    #def gf1check(x,param=None):
        #return np.exp(x[0])+np.exp(x[1])+np.exp(x[2])+np.exp(x[3])-5*x[-1]
    #def dgdq1check(x,param=None):
        #return [np.exp(x[0]),np.exp(x[1]),np.exp(x[2]),np.exp(x[3]),-5.]
    #rvnamescheck = ['r1', 'r2', 'r4', 'r5', 'h']
    #rvscheck = [logr2, logr1, logr4, logr5, rvs[-2]]
    #corrcheck = np.array([[1., 0.3, 0.3, 0.3, 0],
                          #[0.3, 1., 0.3, 0.3, 0],
                          #[0.3, 0.3, 1., 0.3, 0],
                          #[0.3, 0.3, 0.3, 1., 0],
                          #[0,   0,   0,  0,  1.]])
    #probdata = ProbData(names=rvnamescheck, rvs=rvscheck, corr=corrcheck, nataf=False)
    #analysisopt = AnalysisOpt(gradflag='DDM', recordu=False, recordx=False, flagsens=False, verbose=False)
    #gfunc = Gfunc(gf1check, dgdq1check)
    #formBeta = CompReliab(probdata, gfunc, analysisopt)
    #formresults = formBeta.form_result()
    #betacheck = formresults.beta1
    #alphacheck = formresults.alpha

    # limit state 1
    def gf1(x, param=None):
        z = np.sqrt(1.-rolnR)*x[1:-1]+np.sqrt(rolnR)*x[0]
        r = np.exp(logmean+z*logstd)
        return r[0]+r[1]+r[2]+r[3]-5*x[-1]
    def dgdq1(x, param=None):
        z = np.sqrt(1.-rolnR)*x[1:-1]+np.sqrt(rolnR)*x[0]
        r = np.exp(logmean+z*logstd)
        return [np.sqrt(rolnR)*logstd*np.sum(r), r[0]*logstd*np.sqrt(rolnR),
                r[1]*logstd*np.sqrt(rolnR), r[2]*logstd*np.sqrt(rolnR),
                r[3]*logstd*np.sqrt(rolnR), -5.]
    rvnames1 = [rvnames[0], rvnames[1], rvnames[2], rvnames[4], rvnames[5], rvnames[6]]
    rvs1 = [rvs[0], rvs[1], rvs[2], rvs[4], rvs[5], rvs[6]]
    corr1 = np.eye(len(rvnames1))
    probdata1 = ProbData(names=rvnames1, rvs=rvs1, corr=corr1, nataf=False)
    analysisopt1 = AnalysisOpt(gradflag='DDM', recordu=False, recordx=False, flagsens=False, verbose=False)
    gfunc1 = Gfunc(gf1, dgdq1)
    formBeta1 = CompReliab(probdata1, gfunc1, analysisopt1)
    #formresults1 = formBeta1.form_result()
    #beta1 = formresults1.beta1
    #alpha1 = formresults1.alpha

    # limit state 2
    def gf2(x, param=None):
        #x = np.asarray(x)
        z = np.sqrt(1.-rolnR)*x[1:-1]+np.sqrt(rolnR)*x[0]
        r = np.exp(logmean+z*logstd)
        return r[0]+2*r[1]+r[2]-5*x[-1]
    def dgdq2(x, param=None):
        #x = np.asarray(x)
        z = np.sqrt(1.-rolnR)*x[1:-1]+np.sqrt(rolnR)*x[0]
        r = np.exp(logmean+z*logstd)
        return [np.sqrt(rolnR)*logstd*(r[0]+2*r[1]+r[2]), r[0]*logstd*np.sqrt(rolnR),
                2*r[1]*logstd*np.sqrt(rolnR), r[2]*logstd*np.sqrt(rolnR),-5.]
    rvnames2 = [rvnames[0], rvnames[2], rvnames[3], rvnames[4], rvnames[7]]
    rvs2 = [rvs[0], rvs[2], rvs[3], rvs[4], rvs[7]]
    corr2 = np.eye(len(rvnames2))
    probdata2 = ProbData(names=rvnames2, rvs=rvs2, corr=corr2, nataf=False)
    analysisopt2 = AnalysisOpt(gradflag='DDM', recordu=False, recordx=False, flagsens=False, verbose=False)
    gfunc2 = Gfunc(gf2, dgdq2)
    formBeta2 = CompReliab(probdata2, gfunc2, analysisopt2)

    # limit state 3
    def gf3(x, param=None):
        #x = np.asarray(x)
        z = np.sqrt(1.-rolnR)*x[1:-2]+np.sqrt(rolnR)*x[0]
        r = np.exp(logmean+z*logstd)
        return r[0]+2*r[1]+2*r[2]+r[3]-5*x[-2]-5*x[-1]
    def dgdq3(x, param=None):
        #x = np.asarray(x)
        z = np.sqrt(1.-rolnR)*x[1:-2]+np.sqrt(rolnR)*x[0]
        r = np.exp(logmean+z*logstd)
        return [np.sqrt(rolnR)*logstd*(r[0]+2*r[1]+2*r[2]+r[3]), r[0]*logstd*np.sqrt(rolnR),
                2*r[1]*logstd*np.sqrt(rolnR), 2*r[2]*logstd*np.sqrt(rolnR),
                r[3]*logstd*np.sqrt(rolnR), -5., -5.]
    rvnames3 = [rvnames[0], rvnames[1], rvnames[3], rvnames[4], rvnames[5], rvnames[6], rvnames[7]]
    rvs3 = [rvs[0], rvs[1], rvs[3], rvs[4], rvs[5], rvs[6], rvs[7]]
    corr3 = np.eye(len(rvnames3))
    probdata3 = ProbData(names=rvnames3, rvs=rvs3, corr=corr3, nataf=False)
    analysisopt3 = AnalysisOpt(gradflag='DDM', recordu=False, recordx=False, flagsens=False, verbose=False)
    gfunc3 = Gfunc(gf3, dgdq3)
    formBeta3 = CompReliab(probdata3, gfunc3, analysisopt3)

    # system reliability
    sysBeta = SysReliab([formBeta1, formBeta2, formBeta3], [-3])
    sysBeta.set_nCSrv(1)
    sysformres = sysBeta.direct_msr()
    return sysformres.pf


# analytical solution for CTPs from r5 to r4 (deprecated)
def analy2r4(r5bounds, r4bounds, rolnR):
    pdenominator = stats.norm.cdf(r5bounds[-1])-stats.norm.cdf(r5bounds[0])
    syscorr = np.array([[1., -1., rolnR, -rolnR],
                        [-1., 1., -rolnR, rolnR],
                        [rolnR, -rolnR, 1., -1.],
                        [-rolnR, rolnR, -1., 1.]])
    beta = np.array([-r5bounds[0], r5bounds[1], -r4bounds[0], r4bounds[1]])
    mask = np.logical_not(np.logical_or(np.isposinf(beta), np.isneginf(beta)))
    beta = beta[mask]
    syscorr = syscorr[mask,:][:,mask]
    syssize = np.size(beta)
    if syssize ==1:
        pf = stats.norm.cdf(-beta)
        pnominator = 1.-pf
    else:
        sysBeta = SysReliab([], [-np.size(beta)], beta=beta, syscorr=syscorr)
        sysBeta.set_nCSrv(nmax=3)
        sysformres = sysBeta.direct_msr()
        pf = sysformres.pf
        pnominator = 1.-pf
    cpt = pnominator/pdenominator
    return cpt


# FORM from Ri to Mi
def form2m(rvnames, rvs, corr, bins, nstd=200):
    probdata = ProbData(names=rvnames, rvs=rvs, corr=corr, nataf=False)
    def gf1(x, param=None):
        return x[0]+x[1]-bins[0]
    def gf2(x, param=None):
        return bins[-1]-x[0]-x[1]
    def dg1dq(x, param=None):
        dgd1 = 1.
        dgd2 = 1.
        return [dgd1, dgd2]
    def dg2dq(x, param=None):
        dgd1 = -1.
        dgd2 = -1.
        return [dgd1, dgd2]
    analysisopt = AnalysisOpt(gradflag='DDM', recordu=False, recordx=False, flagsens=False, verbose=False)
    gfunc1 = Gfunc(gf1,dg1dq)
    gfunc2 = Gfunc(gf2,dg2dq)
    formBeta1 = CompReliab(probdata, gfunc1, analysisopt)
    formBeta2 = CompReliab(probdata, gfunc2, analysisopt)
    # get rid of numerical issues
    mmean = float(rvs[1].stats('m')); mstd = np.sqrt(float(rvs[1].stats('v')))
    rmean = float(rvs[0].stats('m')); rstd = np.sqrt(float(rvs[0].stats('v')))
    if bins[1]<rmean-nstd*rstd or bins[0]>rmean+nstd*rstd:
        pf = 1.
    else:
        try:
            # compute pf
            if np.isneginf(bins[0]):
                formres = formBeta2.form_result()
                pf = formres.pf1
            elif np.isposinf(bins[-1]):
                formres = formBeta1.form_result()
                pf = formres.pf1
            else:
                sysBeta = SysReliab([formBeta1, formBeta2], [-2])
                sysBeta.set_nCSrv(1)
                sysres = sysBeta.direct_msr()
                pf = sysres.pf
        except np.linalg.LinAlgError:
            # very probably due to low likelihood
            pf = 1.
    return 1.-pf


# deprecated
def msr2e_deprecated(rvnames, rvs, logmean, logstd, rolnR, pbounds):
    # limit state 1
    def gf1(x, param=None):
        z = np.sqrt(1.-rolnR)*x[1:-1]+np.sqrt(rolnR)*x[0]
        r = np.exp(logmean+z*logstd)
        return r[0]+r[1]+r[2]+r[3]-5*x[-1]
    def dgdq1(x, param=None):
        z = np.sqrt(1.-rolnR)*x[1:-1]+np.sqrt(rolnR)*x[0]
        r = np.exp(logmean+z*logstd)
        return [np.sqrt(rolnR)*logstd*np.sum(r), r[0]*logstd*np.sqrt(rolnR),
                r[1]*logstd*np.sqrt(rolnR), r[2]*logstd*np.sqrt(rolnR),
                r[3]*logstd*np.sqrt(rolnR), -5.]
    rvnames1 = [rvnames[0], rvnames[1], rvnames[2], rvnames[4], rvnames[5], rvnames[6]]
    rvs1 = [rvs[0], rvs[1], rvs[2], rvs[4], rvs[5], rvs[6]]
    corr1 = np.eye(len(rvnames1))
    probdata1 = ProbData(names=rvnames1, rvs=rvs1, corr=corr1, nataf=False)
    analysisopt1 = AnalysisOpt(gradflag='DDM', recordu=False, recordx=False, flagsens=False, verbose=False)
    gfunc1 = Gfunc(gf1, dgdq1)
    formBeta1 = CompReliab(probdata1, gfunc1, analysisopt1)

    # limit state 2
    def gf2(x, param=None):
        z = np.sqrt(1.-rolnR)*x[1:-1]+np.sqrt(rolnR)*x[0]
        r = np.exp(logmean+z*logstd)
        return r[0]+2*r[1]+r[2]-5*x[-1]
    def dgdq2(x, param=None):
        z = np.sqrt(1.-rolnR)*x[1:-1]+np.sqrt(rolnR)*x[0]
        r = np.exp(logmean+z*logstd)
        return [np.sqrt(rolnR)*logstd*(r[0]+2*r[1]+r[2]), r[0]*logstd*np.sqrt(rolnR),
                2*r[1]*logstd*np.sqrt(rolnR), r[2]*logstd*np.sqrt(rolnR),-5.]
    rvnames2 = [rvnames[0], rvnames[2], rvnames[3], rvnames[4], rvnames[7]]
    rvs2 = [rvs[0], rvs[2], rvs[3], rvs[4], rvs[7]]
    corr2 = np.eye(len(rvnames2))
    probdata2 = ProbData(names=rvnames2, rvs=rvs2, corr=corr2, nataf=False)
    analysisopt2 = AnalysisOpt(gradflag='DDM', recordu=False, recordx=False, flagsens=False, verbose=False)
    gfunc2 = Gfunc(gf2, dgdq2)
    formBeta2 = CompReliab(probdata2, gfunc2, analysisopt2)

    # limit state 3
    def gf3(x, param=None):
        z = np.sqrt(1.-rolnR)*x[1:-2]+np.sqrt(rolnR)*x[0]
        r = np.exp(logmean+z*logstd)
        return r[0]+2*r[1]+2*r[2]+r[3]-5*x[-2]-5*x[-1]
    def dgdq3(x, param=None):
        z = np.sqrt(1.-rolnR)*x[1:-2]+np.sqrt(rolnR)*x[0]
        r = np.exp(logmean+z*logstd)
        return [np.sqrt(rolnR)*logstd*(r[0]+2*r[1]+2*r[2]+r[3]), r[0]*logstd*np.sqrt(rolnR),
                2*r[1]*logstd*np.sqrt(rolnR), 2*r[2]*logstd*np.sqrt(rolnR),
                r[3]*logstd*np.sqrt(rolnR), -5., -5.]
    rvnames3 = [rvnames[0], rvnames[1], rvnames[3], rvnames[4], rvnames[5], rvnames[6], rvnames[7]]
    rvs3 = [rvs[0], rvs[1], rvs[3], rvs[4], rvs[5], rvs[6], rvs[7]]
    corr3 = np.eye(len(rvnames3))
    probdata3 = ProbData(names=rvnames3, rvs=rvs3, corr=corr3, nataf=False)
    analysisopt3 = AnalysisOpt(gradflag='DDM', recordu=False, recordx=False, flagsens=False, verbose=False)
    gfunc3 = Gfunc(gf3, dgdq3)
    formBeta3 = CompReliab(probdata3, gfunc3, analysisopt3)

    # limit state 4
    def gf4(x, param=None):
        z = np.sqrt(1.-rolnR)*x[1]+np.sqrt(rolnR)*x[0]
        r4 = np.exp(logmean+z*logstd)
        return r4-pbounds[0,0]
    def dgdq4(x, param=None):
        z = np.sqrt(1.-rolnR)*x[1]+np.sqrt(rolnR)*x[0]
        r4 = np.exp(logmean+z*logstd)
        return [np.sqrt(rolnR)*logstd*(r4), r4*logstd*np.sqrt(rolnR)]
    rvnames4 = [rvnames[0], rvnames[4]]
    rvs4 = [rvs[0], rvs[4]]
    corr4 = np.eye(len(rvnames4))
    probdata4 = ProbData(names=rvnames4, rvs=rvs4, corr=corr4, nataf=False)
    analysisopt4 = AnalysisOpt(gradflag='DDM', recordu=False, recordx=False, flagsens=False, verbose=False)
    gfunc4 = Gfunc(gf4, dgdq4)
    formBeta4 = CompReliab(probdata4, gfunc4, analysisopt4)

    # limit state 5
    def gf5(x, param=None):
        z = np.sqrt(1.-rolnR)*x[1]+np.sqrt(rolnR)*x[0]
        r4 = np.exp(logmean+z*logstd)
        return pbounds[0,1]-r4
    def dgdq5(x, param=None):
        z = np.sqrt(1.-rolnR)*x[1]+np.sqrt(rolnR)*x[0]
        r4 = np.exp(logmean+z*logstd)
        return [-np.sqrt(rolnR)*logstd*(r4), -r4*logstd*np.sqrt(rolnR)]
    rvnames5 = [rvnames[0], rvnames[4]]
    rvs5 = [rvs[0], rvs[4]]
    corr5 = np.eye(len(rvnames5))
    probdata5 = ProbData(names=rvnames5, rvs=rvs5, corr=corr5, nataf=False)
    analysisopt5 = AnalysisOpt(gradflag='DDM', recordu=False, recordx=False, flagsens=False, verbose=False)
    gfunc5 = Gfunc(gf5, dgdq5)
    formBeta5 = CompReliab(probdata5, gfunc5, analysisopt5)

    # limit state 6
    def gf6(x, param=None):
        z = np.sqrt(1.-rolnR)*x[1]+np.sqrt(rolnR)*x[0]
        r5 = np.exp(logmean+z*logstd)
        return r5-pbounds[1,0]
    def dgdq6(x, param=None):
        z = np.sqrt(1.-rolnR)*x[1]+np.sqrt(rolnR)*x[0]
        r5 = np.exp(logmean+z*logstd)
        return [np.sqrt(rolnR)*logstd*(r5), r5*logstd*np.sqrt(rolnR)]
    rvnames6 = [rvnames[0], rvnames[5]]
    rvs6 = [rvs[0], rvs[5]]
    corr6 = np.eye(len(rvnames6))
    probdata6 = ProbData(names=rvnames6, rvs=rvs6, corr=corr6, nataf=False)
    analysisopt6 = AnalysisOpt(gradflag='DDM', recordu=False, recordx=False, flagsens=False, verbose=False)
    gfunc6 = Gfunc(gf6, dgdq6)
    formBeta6 = CompReliab(probdata6, gfunc6, analysisopt6)

    # limit state 7
    def gf7(x, param=None):
        z = np.sqrt(1.-rolnR)*x[1]+np.sqrt(rolnR)*x[0]
        r5 = np.exp(logmean+z*logstd)
        return pbounds[1,1]-r5
    def dgdq7(x, param=None):
        z = np.sqrt(1.-rolnR)*x[1]+np.sqrt(rolnR)*x[0]
        r5 = np.exp(logmean+z*logstd)
        return [-np.sqrt(rolnR)*logstd*(r5), -r5*logstd*np.sqrt(rolnR)]
    rvnames7 = [rvnames[0], rvnames[5]]
    rvs7 = [rvs[0], rvs[5]]
    corr7 = np.eye(len(rvnames7))
    probdata7 = ProbData(names=rvnames7, rvs=rvs7, corr=corr7, nataf=False)
    analysisopt7 = AnalysisOpt(gradflag='DDM', recordu=False, recordx=False, flagsens=False, verbose=False)
    gfunc7 = Gfunc(gf7, dgdq7)
    formBeta7 = CompReliab(probdata7, gfunc7, analysisopt7)

    # system reliability
    flatbounds = pbounds.flatten()
    boundlimits = np.array([formBeta4, formBeta5, formBeta6, formBeta7])
    mask = np.logical_not(np.logical_or(np.isposinf(flatbounds), np.isneginf(flatbounds)))
    usedlimits = boundlimits[mask]
    syslimits = np.hstack(([formBeta1, formBeta2, formBeta3],usedlimits))
    sysBeta = SysReliab(syslimits, [-np.size(syslimits)])
    sysBeta.set_nCSrv()
    sysformres = sysBeta.direct_msr()
    return sysformres.pf


def msr2e(rvnames, rvs, logmean, logstd, rolnR):
    # limit state 1
    def gf1(x, param=None):
        z = np.sqrt(1.-rolnR)*x[1:3]+np.sqrt(rolnR)*x[0]
        r = np.exp(logmean+z*logstd)
        return r[0]+r[1]+x[3]+x[4]-5*x[-1]
    def dgdq1(x, param=None):
        z = np.sqrt(1.-rolnR)*x[1:3]+np.sqrt(rolnR)*x[0]
        r = np.exp(logmean+z*logstd)
        return [np.sqrt(rolnR)*logstd*(r[0]+r[1]), r[0]*logstd*np.sqrt(rolnR),
                r[1]*logstd*np.sqrt(rolnR), 1., 1., -5.]
    rvnames1 = [rvnames[0], rvnames[1], rvnames[2], rvnames[4], rvnames[5], rvnames[6]]
    rvs1 = [rvs[0], rvs[1], rvs[2], rvs[4], rvs[5], rvs[6]]
    corr1 = np.eye(len(rvnames1))
    probdata1 = ProbData(names=rvnames1, rvs=rvs1, corr=corr1, nataf=False)
    analysisopt1 = AnalysisOpt(gradflag='DDM', recordu=False, recordx=False, flagsens=False, verbose=False)
    gfunc1 = Gfunc(gf1, dgdq1)
    formBeta1 = CompReliab(probdata1, gfunc1, analysisopt1)

    # limit state 2
    def gf2(x, param=None):
        z = np.sqrt(1.-rolnR)*x[1:3]+np.sqrt(rolnR)*x[0]
        r = np.exp(logmean+z*logstd)
        return r[0]+2*r[1]+x[3]-5*x[-1]
    def dgdq2(x, param=None):
        z = np.sqrt(1.-rolnR)*x[1:3]+np.sqrt(rolnR)*x[0]
        r = np.exp(logmean+z*logstd)
        return [np.sqrt(rolnR)*logstd*(r[0]+2*r[1]), r[0]*logstd*np.sqrt(rolnR),
                2*r[1]*logstd*np.sqrt(rolnR), 1., -5.]
    rvnames2 = [rvnames[0], rvnames[2], rvnames[3], rvnames[4], rvnames[7]]
    rvs2 = [rvs[0], rvs[2], rvs[3], rvs[4], rvs[7]]
    corr2 = np.eye(len(rvnames2))
    probdata2 = ProbData(names=rvnames2, rvs=rvs2, corr=corr2, nataf=False)
    analysisopt2 = AnalysisOpt(gradflag='DDM', recordu=False, recordx=False, flagsens=False, verbose=False)
    gfunc2 = Gfunc(gf2, dgdq2)
    formBeta2 = CompReliab(probdata2, gfunc2, analysisopt2)

    # limit state 3
    def gf3(x, param=None):
        z = np.sqrt(1.-rolnR)*x[1:3]+np.sqrt(rolnR)*x[0]
        r = np.exp(logmean+z*logstd)
        return r[0]+2*r[1]+2*x[3]+x[4]-5*x[-2]-5*x[-1]
    def dgdq3(x, param=None):
        z = np.sqrt(1.-rolnR)*x[1:3]+np.sqrt(rolnR)*x[0]
        r = np.exp(logmean+z*logstd)
        return [np.sqrt(rolnR)*logstd*(r[0]+2*r[1]), r[0]*logstd*np.sqrt(rolnR),
                2*r[1]*logstd*np.sqrt(rolnR), 2., 1., -5., -5.]
    rvnames3 = [rvnames[0], rvnames[1], rvnames[3], rvnames[4], rvnames[5], rvnames[6], rvnames[7]]
    rvs3 = [rvs[0], rvs[1], rvs[3], rvs[4], rvs[5], rvs[6], rvs[7]]
    corr3 = np.eye(len(rvnames3))
    probdata3 = ProbData(names=rvnames3, rvs=rvs3, corr=corr3, nataf=False)
    analysisopt3 = AnalysisOpt(gradflag='DDM', recordu=False, recordx=False, flagsens=False, verbose=False)
    gfunc3 = Gfunc(gf3, dgdq3)
    formBeta3 = CompReliab(probdata3, gfunc3, analysisopt3)

    # system reliability
    try:
        #import ipdb; ipdb.set_trace() # BREAKPOINT
        #debug = formBeta1.form_result().pf1
        sysBeta = SysReliab([formBeta1, formBeta2, formBeta3], [-3])
        sysBeta.set_nCSrv()
        sysformres = sysBeta.direct_msr()
        pf = sysformres.pf
    except np.linalg.LinAlgError:
        pf = 0.
    return pf