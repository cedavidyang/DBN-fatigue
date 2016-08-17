import os
# strepath = os.path.join(os.path.expanduser('~'), 'FERUM')
strepath = os.path.abspath('../')
import sys
sys.path.append(strepath)
import scipy.integrate as intg

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


# from Ri to Mi
def msr2m(rvnames, rvs, logmean, logstd, trunclb, truncub, mbins):
    rlb = trunclb[0]
    rub = truncub[0]
    # if np.isinf(rub): rub = np.exp(logmean+10*logstd)
    ulb = (np.log(rlb)-logmean)/logstd
    uub = (np.log(rub)-logmean)/logstd
    # if np.isinf(ulb): ulb = -10.
    # if np.isneginf(mbins[0]): mbins[0] = np.exp(logmean-20*logstd)
    # if np.isposinf(mbins[1]): mbins[1] = np.exp(logmean+20*logstd)
    corr = np.eye(len(rvnames))
    probdata = ProbData(names=rvnames, rvs=rvs, corr=corr, nataf=False)
    analysisopt = AnalysisOpt(gradflag='DDM', recordu=False, recordx=False,
            flagsens=False, verbose=False)

    # limit state 1
    def gf1(x, param=None):
        m = np.exp(logmean+x[0]*logstd) + x[1]
        return m-mbins[0]
    def dg1dq(x, param=None):
        dgd1 = np.exp(logmean+x[0]*logstd)*logstd
        dgd2 = 1.
        return [dgd1, dgd2]
    gfunc1 = Gfunc(gf1,dg1dq)
    formBeta1 = CompReliab(probdata, gfunc1, analysisopt)

    # limit state 2
    def gf2(x, param=None):
        m = np.exp(logmean+x[0]*logstd) + x[1]
        return mbins[1]-m
    def dg2dq(x, param=None):
        dgd1 = -np.exp(logmean+x[0]*logstd)*logstd
        dgd2 = -1.
        return [dgd1, dgd2]
    gfunc2 = Gfunc(gf2,dg2dq)
    formBeta2 = CompReliab(probdata, gfunc2, analysisopt)

    # limit state 3
    def gf3(x, param=None):
        return x[0]-ulb
    def dg3dq(x, param=None):
        return [1., 0.]
    gfunc3 = Gfunc(gf3,dg3dq)
    formBeta3 = CompReliab(probdata, gfunc3, analysisopt)

    # limit state 4
    def gf4(x, param=None):
        return uub-x[0]
    def dg4dq(x, param=None):
        return [-1., 0.]
    gfunc4 = Gfunc(gf4,dg4dq)
    formBeta4 = CompReliab(probdata, gfunc4, analysisopt)

    try:
        if np.isneginf(ulb):
            if np.isneginf(mbins[0]):
                sysBeta = SysReliab([formBeta2, formBeta4], [-2])
                sysformres = sysBeta.mvn_msr(sysBeta.syscorr)
            elif np.isposinf(mbins[1]):
                sysBeta = SysReliab([formBeta1, formBeta4], [-2])
                sysformres = sysBeta.mvn_msr(sysBeta.syscorr)
            else:
                sysBeta = SysReliab([formBeta1, formBeta2, formBeta4], [-3])
                sysformres = sysBeta.mvn_msr(sysBeta.syscorr)
        elif np.isposinf(uub):
            if np.isneginf(mbins[0]):
                sysBeta = SysReliab([formBeta2, formBeta3], [-2])
                sysformres = sysBeta.mvn_msr(sysBeta.syscorr)
            elif np.isposinf(mbins[1]):
                sysBeta = SysReliab([formBeta1, formBeta3], [-2])
                sysformres = sysBeta.mvn_msr(sysBeta.syscorr)
            else:
                sysBeta = SysReliab([formBeta1, formBeta2, formBeta3], [-3])
                sysformres = sysBeta.mvn_msr(sysBeta.syscorr)
        else:
            if np.isneginf(mbins[0]):
                sysBeta = SysReliab([formBeta2, formBeta3, formBeta4], [-3])
                sysformres = sysBeta.mvn_msr(sysBeta.syscorr)
            elif np.isposinf(mbins[1]):
                sysBeta = SysReliab([formBeta1, formBeta3, formBeta4], [-3])
                sysformres = sysBeta.mvn_msr(sysBeta.syscorr)
            else:
                sysBeta = SysReliab([formBeta1, formBeta2, formBeta3, formBeta4], [-4])
                sysformres = sysBeta.mvn_msr(sysBeta.syscorr)
        ptotal = 1.-sysformres.pf
    except np.linalg.LinAlgError:
        ptotal = 0.
    pcond = rvs[0].cdf(uub)-rvs[0].cdf(ulb)
    p = ptotal/pcond
    return p


# from Ri to E
def msr2e(rvnames, rvs, logmean, logstd, rolnR, trunclb, truncub):
    r4lb = trunclb[0]
    r4ub = truncub[0]
    if np.isinf(r4ub): r4ub = np.exp(logmean+10*logstd)
    r5lb = trunclb[1]
    r5ub = truncub[1]
    if np.isinf(r5ub): r5ub = np.exp(logmean+10*logstd)
    corr = np.eye(len(rvnames))
    probdata = ProbData(names=rvnames, rvs=rvs, corr=corr, nataf=False)
    analysisopt = AnalysisOpt(gradflag='DDM', recordu=False, recordx=False,
            flagsens=False, verbose=False)
    # limit state 1
    def gf1(x, param=None):
        z = np.sqrt(1.-rolnR)*x[1:-2]+np.sqrt(rolnR)*x[0]
        r = np.exp(logmean+z*logstd)
        return r[0]+r[1]+r[3]+r[4]-5*x[-2]
    def dgdq1(x, param=None):
        z = np.sqrt(1.-rolnR)*x[1:-2]+np.sqrt(rolnR)*x[0]
        r = np.exp(logmean+z*logstd)
        return [np.sqrt(rolnR)*logstd*(r[0]+r[1]+r[3]+r[4]),
                r[0]*logstd*np.sqrt(1.-rolnR),
                r[1]*logstd*np.sqrt(1.-rolnR),
                0.,
                r[3]*logstd*np.sqrt(1.-rolnR),
                r[4]*logstd*np.sqrt(1.-rolnR),
                -5., 0.]
    gfunc1 = Gfunc(gf1, dgdq1)
    formBeta1 = CompReliab(probdata, gfunc1, analysisopt)

    # limit state 2
    def gf2(x, param=None):
        z = np.sqrt(1.-rolnR)*x[1:-2]+np.sqrt(rolnR)*x[0]
        r = np.exp(logmean+z*logstd)
        return r[1]+2*r[2]+r[3]-5*x[-1]
    def dgdq2(x, param=None):
        z = np.sqrt(1.-rolnR)*x[1:-2]+np.sqrt(rolnR)*x[0]
        r = np.exp(logmean+z*logstd)
        return [np.sqrt(rolnR)*logstd*(r[1]+2*r[2]+r[3]),
                0.,
                r[1]*logstd*np.sqrt(1.-rolnR),
                2*r[2]*logstd*np.sqrt(1.-rolnR),
                r[3]*logstd*np.sqrt(1.-rolnR),
                0., 0., -5.]
    gfunc2 = Gfunc(gf2, dgdq2)
    formBeta2 = CompReliab(probdata, gfunc2, analysisopt)

    # limit state 3
    def gf3(x, param=None):
        z = np.sqrt(1.-rolnR)*x[1:-2]+np.sqrt(rolnR)*x[0]
        r = np.exp(logmean+z*logstd)
        return r[0]+2*r[2]+2*r[3]+r[4]-5*x[-2]-5*x[-1]
    def dgdq3(x, param=None):
        z = np.sqrt(1.-rolnR)*x[1:-2]+np.sqrt(rolnR)*x[0]
        r = np.exp(logmean+z*logstd)
        return [np.sqrt(rolnR)*logstd*(r[0]+2*r[2]+2*r[3]+r[4]),
                r[0]*logstd*np.sqrt(1.-rolnR),
                0.,
                2*r[2]*logstd*np.sqrt(1.-rolnR),
                2*r[3]*logstd*np.sqrt(1.-rolnR),
                r[4]*logstd*np.sqrt(1.-rolnR),
                -5., -5.]
    gfunc3 = Gfunc(gf3, dgdq3)
    formBeta3 = CompReliab(probdata, gfunc3, analysisopt)

    # limit state 4
    def gf4(x, param=None):
        z = np.sqrt(1.-rolnR)*x[1:-2]+np.sqrt(rolnR)*x[0]
        r = np.exp(logmean+z*logstd)
        return r[3]-r4lb
    def dgdq4(x, param=None):
        z = np.sqrt(1.-rolnR)*x[1:-2]+np.sqrt(rolnR)*x[0]
        r = np.exp(logmean+z*logstd)
        return [np.sqrt(rolnR)*logstd*r[3],
                0.,
                0.,
                0.,
                r[3]*logstd*np.sqrt(1.-rolnR),
                0.,0.,0.]
    gfunc4 = Gfunc(gf4, dgdq4)
    formBeta4 = CompReliab(probdata, gfunc4, analysisopt)


    # limit state 5
    def gf5(x, param=None):
        z = np.sqrt(1.-rolnR)*x[1:-2]+np.sqrt(rolnR)*x[0]
        r = np.exp(logmean+z*logstd)
        return r4ub-r[3]
    def dgdq5(x, param=None):
        z = np.sqrt(1.-rolnR)*x[1:-2]+np.sqrt(rolnR)*x[0]
        r = np.exp(logmean+z*logstd)
        return [-np.sqrt(rolnR)*logstd*r[3],
                0.,
                0.,
                0.,
                -r[3]*logstd*np.sqrt(1.-rolnR),
                0.,0.,0.]
    gfunc5 = Gfunc(gf5, dgdq5)
    formBeta5 = CompReliab(probdata, gfunc5, analysisopt)

    # limit state 6
    def gf6(x, param=None):
        z = np.sqrt(1.-rolnR)*x[1:-2]+np.sqrt(rolnR)*x[0]
        r = np.exp(logmean+z*logstd)
        return r[4]-r5lb
    def dgdq6(x, param=None):
        z = np.sqrt(1.-rolnR)*x[1:-2]+np.sqrt(rolnR)*x[0]
        r = np.exp(logmean+z*logstd)
        return [np.sqrt(rolnR)*logstd*r[4],
                0.,
                0.,
                0.,
                0.,
                r[4]*logstd*np.sqrt(1.-rolnR),
                0.,0.]
    gfunc6 = Gfunc(gf6, dgdq6)
    formBeta6 = CompReliab(probdata, gfunc6, analysisopt)


    # limit state 7
    def gf7(x, param=None):
        z = np.sqrt(1.-rolnR)*x[1:-2]+np.sqrt(rolnR)*x[0]
        r = np.exp(logmean+z*logstd)
        return r5ub-r[4]
    def dgdq7(x, param=None):
        z = np.sqrt(1.-rolnR)*x[1:-2]+np.sqrt(rolnR)*x[0]
        r = np.exp(logmean+z*logstd)
        return [-np.sqrt(rolnR)*logstd*r[4],
                0.,
                0.,
                0.,
                0.,
                -r[4]*logstd*np.sqrt(1.-rolnR),
                0.,0.]
    gfunc7 = Gfunc(gf7, dgdq7)
    formBeta7 = CompReliab(probdata, gfunc7, analysisopt)

    # system reliability
    try:
        sysBeta = SysReliab([formBeta1, formBeta2, formBeta3,
            formBeta4, formBeta5, formBeta6, formBeta7], [-7])
        sysBetacond = SysReliab([formBeta4, formBeta5, formBeta6, formBeta7], [-4])
        sysformres = sysBeta.mvn_msr(sysBeta.syscorr)
        sysformrescond = sysBetacond.mvn_msr(sysBetacond.syscorr)
        ptotal_safe = 1.-sysformres.pf
        pcond = 1.-sysformrescond.pf
        pf = 1.-ptotal_safe/pcond
        while pf<0:
            sysformres = sysBeta.mvn_msr(sysBeta.syscorr)
            sysformrescond = sysBetacond.mvn_msr(sysBetacond.syscorr)
            ptotal_safe = 1.-sysformres.pf
            pcond = 1.-sysformrescond.pf
            pf = 1.-ptotal_safe/pcond
    except np.linalg.LinAlgError:
        pf = 0.
    return pf


# from Ri to E by MC
def msr2e_mc(rvnames, rvs, logmean, logstd, rolnR, truncrvs, nsmp=int(1e7)):
    r4lb = trunclb[0]
    r4ub = truncub[0]
    if np.isinf(r4ub): r4ub = np.exp(logmean+10*logstd)
    r5lb = trunclb[1]
    r5ub = truncub[1]
    if np.isinf(r5ub): r5ub = np.exp(logmean+10*logstd)
    ur = rvs[0].rvs(size=nsmp)
    u1 = rvs[1].rvs(size=nsmp)
    u2 = rvs[2].rvs(size=nsmp)
    u3 = rvs[3].rvs(size=nsmp)
    u4 = rvs[4].rvs(size=nsmp)
    u5 = rvs[4].rvs(size=nsmp)
    h = rvs[6].rvs(size=nsmp)
    v = rvs[7].rvs(size=nsmp)
    r15 = np.empty(5, dtype=object)
    for i,ui in enumerate([u1,u2,u3,u4,u5]):
        zi = np.sqrt(1.-rolnR)*ui+np.sqrt(rolnR)*ur
        r15[i] = np.exp(logmean+zi*logstd)
    r1 = r15[0]; r2 = r15[1]; r3=r15[2]; r4=r15[3]; r5=r15[4]
    g1 = r1+r2+r4+r5-5*h
    g2 = r2+2*r3+r4-5*v
    g3 = r1+2*r3+2*r4+r5-5*h-5*v
    g4 = (r4-r4lb)*(r4-r4ub)
    g5 = (r5-r5lb)*(r5-r5ub)
    gsyssmp = np.min([g1>=0, g2>=0, g3>=0, g4<=0, g5<=0], axis=0)
    gcond = np.min([g4<=0, g5<=0], axis=0)
    pf = 1.-np.sum(gsyssmp,dtype=float)/np.sum(gcond,dtype=float)
    if np.isnan(pf) or np.isinf(pf):
        pf = 0.
    return pf


# from Ri to E
def msr2q(rvnames, rvs, logmean, logstd, rolnR, trunclb, truncub, qbin):
    r4lb = trunclb[0]
    r4ub = truncub[0]
    if np.isinf(r4ub): r4ub = np.exp(logmean+10*logstd)
    r5lb = trunclb[1]
    r5ub = truncub[1]
    if np.isinf(r5ub): r5ub = np.exp(logmean+10*logstd)
    qplus = qbin[1]
    corr = np.eye(len(rvnames))
    probdata = ProbData(names=rvnames, rvs=rvs, corr=corr, nataf=False)
    analysisopt = AnalysisOpt(gradflag='DDM', recordu=False, recordx=False,
            flagsens=False, verbose=False)
    # limit state 1
    def gf1(x, param=None):
        z = np.sqrt(1.-rolnR)*x[1:-1]+np.sqrt(rolnR)*x[0]
        r = np.exp(logmean+z*logstd)
        return r[0]+r[1]+r[3]+r[4]-5*qplus
    def dgdq1(x, param=None):
        z = np.sqrt(1.-rolnR)*x[1:-1]+np.sqrt(rolnR)*x[0]
        r = np.exp(logmean+z*logstd)
        return [np.sqrt(rolnR)*logstd*(r[0]+r[1]+r[3]+r[4]),
                r[0]*logstd*np.sqrt(1.-rolnR),
                r[1]*logstd*np.sqrt(1.-rolnR),
                0.,
                r[3]*logstd*np.sqrt(1.-rolnR),
                r[4]*logstd*np.sqrt(1.-rolnR),
                0.]
    gfunc1 = Gfunc(gf1, dgdq1)
    formBeta1 = CompReliab(probdata, gfunc1, analysisopt)

    # limit state 2
    def gf2(x, param=None):
        z = np.sqrt(1.-rolnR)*x[1:-1]+np.sqrt(rolnR)*x[0]
        r = np.exp(logmean+z*logstd)
        return r[1]+2*r[2]+r[3]-5*x[-1]
    def dgdq2(x, param=None):
        z = np.sqrt(1.-rolnR)*x[1:-1]+np.sqrt(rolnR)*x[0]
        r = np.exp(logmean+z*logstd)
        return [np.sqrt(rolnR)*logstd*(r[1]+2*r[2]+r[3]),
                0.,
                r[1]*logstd*np.sqrt(1.-rolnR),
                2*r[2]*logstd*np.sqrt(1.-rolnR),
                r[3]*logstd*np.sqrt(1.-rolnR),
                0., -5.]
    gfunc2 = Gfunc(gf2, dgdq2)
    formBeta2 = CompReliab(probdata, gfunc2, analysisopt)

    # limit state 3
    def gf3(x, param=None):
        z = np.sqrt(1.-rolnR)*x[1:-1]+np.sqrt(rolnR)*x[0]
        r = np.exp(logmean+z*logstd)
        return r[0]+2*r[2]+2*r[3]+r[4]-5*qplus-5*x[-1]
    def dgdq3(x, param=None):
        z = np.sqrt(1.-rolnR)*x[1:-1]+np.sqrt(rolnR)*x[0]
        r = np.exp(logmean+z*logstd)
        return [np.sqrt(rolnR)*logstd*(r[0]+2*r[2]+2*r[3]+r[4]),
                r[0]*logstd*np.sqrt(1.-rolnR),
                0.,
                2*r[2]*logstd*np.sqrt(1.-rolnR),
                2*r[3]*logstd*np.sqrt(1.-rolnR),
                r[4]*logstd*np.sqrt(1.-rolnR),
                -5.]
    gfunc3 = Gfunc(gf3, dgdq3)
    formBeta3 = CompReliab(probdata, gfunc3, analysisopt)

    # limit state 4
    def gf4(x, param=None):
        z = np.sqrt(1.-rolnR)*x[1:-1]+np.sqrt(rolnR)*x[0]
        r = np.exp(logmean+z*logstd)
        return r[3]-r4lb
    def dgdq4(x, param=None):
        z = np.sqrt(1.-rolnR)*x[1:-1]+np.sqrt(rolnR)*x[0]
        r = np.exp(logmean+z*logstd)
        return [np.sqrt(rolnR)*logstd*r[3],
                0.,
                0.,
                0.,
                r[3]*logstd*np.sqrt(1.-rolnR),
                0.,0.]
    gfunc4 = Gfunc(gf4, dgdq4)
    formBeta4 = CompReliab(probdata, gfunc4, analysisopt)


    # limit state 5
    def gf5(x, param=None):
        z = np.sqrt(1.-rolnR)*x[1:-1]+np.sqrt(rolnR)*x[0]
        r = np.exp(logmean+z*logstd)
        return r4ub-r[3]
    def dgdq5(x, param=None):
        z = np.sqrt(1.-rolnR)*x[1:-1]+np.sqrt(rolnR)*x[0]
        r = np.exp(logmean+z*logstd)
        return [-np.sqrt(rolnR)*logstd*r[3],
                0.,
                0.,
                0.,
                -r[3]*logstd*np.sqrt(1.-rolnR),
                0.,0.]
    gfunc5 = Gfunc(gf5, dgdq5)
    formBeta5 = CompReliab(probdata, gfunc5, analysisopt)

    # limit state 6
    def gf6(x, param=None):
        z = np.sqrt(1.-rolnR)*x[1:-1]+np.sqrt(rolnR)*x[0]
        r = np.exp(logmean+z*logstd)
        return r[4]-r5lb
    def dgdq6(x, param=None):
        z = np.sqrt(1.-rolnR)*x[1:-1]+np.sqrt(rolnR)*x[0]
        r = np.exp(logmean+z*logstd)
        return [np.sqrt(rolnR)*logstd*r[4],
                0.,
                0.,
                0.,
                0.,
                r[4]*logstd*np.sqrt(1.-rolnR),
                0.]
    gfunc6 = Gfunc(gf6, dgdq6)
    formBeta6 = CompReliab(probdata, gfunc6, analysisopt)

    # limit state 7
    def gf7(x, param=None):
        z = np.sqrt(1.-rolnR)*x[1:-1]+np.sqrt(rolnR)*x[0]
        r = np.exp(logmean+z*logstd)
        return r5ub-r[4]
    def dgdq7(x, param=None):
        z = np.sqrt(1.-rolnR)*x[1:-1]+np.sqrt(rolnR)*x[0]
        r = np.exp(logmean+z*logstd)
        return [-np.sqrt(rolnR)*logstd*r[4],
                0.,
                0.,
                0.,
                0.,
                -r[4]*logstd*np.sqrt(1.-rolnR),
                0.]
    gfunc7 = Gfunc(gf7, dgdq7)
    formBeta7 = CompReliab(probdata, gfunc7, analysisopt)

    # system reliability
    try:
        sysBeta = SysReliab([formBeta1, formBeta2, formBeta3, formBeta4,
            formBeta5, formBeta6, formBeta7], [-7])
        sysBetacond = SysReliab([formBeta4, formBeta5, formBeta6, formBeta7], [-4])
        sysformres = sysBeta.mvn_msr(sysBeta.syscorr)
        sysformrescond = sysBetacond.mvn_msr(sysBetacond.syscorr)
        ptotal_safe = 1.-sysformres.pf
        pcond = 1.-sysformrescond.pf
        pf = 1.-ptotal_safe/pcond
        while pf<0:
            sysformres = sysBeta.mvn_msr(sysBeta.syscorr)
            sysformrescond = sysBetacond.mvn_msr(sysBetacond.syscorr)
            ptotal_safe = 1.-sysformres.pf
            pcond = 1.-sysformrescond.pf
            pf = 1.-ptotal_safe/pcond
    except np.linalg.LinAlgError:
        pf = 0.
    return pf


def intg2h(uhrv, trunclb, truncub, hbin, abstol=1e-12, reltol=1e-8, nsmp=int(1e6)):
    uhlb = trunclb[0]
    uhub = truncub[0]
    hlb = hbin[0]
    hub = hbin[1]
    # if np.isposinf(uhub): uhub=10*uhlb
    # if np.isposinf(hub): hub=10*hlb
    def integrnd(uh,h,uhrv):
        hmean = uh+9.
        hstd = 20.
        beta = (hstd)/(np.pi/np.sqrt(6))
        mu = hmean-np.euler_gamma*beta
        hrv = stats.gumbel_r(loc=mu, scale=beta)
        pdfproduct = hrv.pdf(h)*uhrv.pdf(uh)
        return pdfproduct
    p,err = intg.dblquad(lambda x,y: integrnd(x,y,uhrv), hlb, hub,
            lambda y: uhlb, lambda y: uhub, epsabs=abstol)
    return p


def intg2h_mc(uhrv, trunclb, truncub, hbin, abstol=1e-12, reltol=1e-8, nsmp=int(1e5)):
    uhlb = trunclb[0]
    uhub = truncub[0]
    hlb = hbin[0]
    hub = hbin[1]
    uhsmp = uhrv.rvs(size=nsmp)
    hmeansmp = uhsmp+9.
    pdfsum = 0.
    for hmean in hmeansmp:
        hstd = 20.
        beta = (hstd)/(np.pi/np.sqrt(6))
        mu = hmean-np.euler_gamma*beta
        hrv = stats.gumbel_r(loc=mu, scale=beta)
        pdfsum += hrv.cdf(hub)-hrv.cdf(hlb)
    p=pdfsum/nsmp
    return p


def intg2e(qrv, hrv, abstol=1e-12, reltol=1e-8):
    qlb = qrv.lb
    qub = qrv.ub
    hlb = hrv.lb
    hub = hrv.ub
    if np.isposinf(hub) or np.isposinf(qub):
        def integrnd(h,q,hrv,qrv):
            res = np.array((h-q)>0,dtype=float)*hrv.pdf(h)*qrv.pdf(q)
            return res
        p,err = intg.dblquad(lambda x,y: integrnd(x,y,hrv,qrv), hlb, hub,
                lambda y: qlb, lambda y: qub, epsabs=abstol)
    else:    # two uniform distributions
        if qub<=hlb:
            p = 1.
        elif qlb>=hub:
            p = 0.
        elif hub>qlb and hlb<=qlb:
            p = 0.5*(hub-qlb)**2/((hub-hlb)*(qub-qlb))
        elif hub>qub and hlb<=qub:
            p = 1. - 0.5*(qub-hlb)**2/((hub-hlb)*(qub-qlb))
        else:
            print 'sth is wrong'
            import ipdb; ipdb.set_trace() # BREAKPOINT
    return p
