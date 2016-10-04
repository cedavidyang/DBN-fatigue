import os
strepath = os.path.abspath('../')
import sys
sys.path.append(strepath)
import scipy.integrate as intg
import scipy.optimize as opt
from scipy.special import gamma

from pyStRe import *
import numpy as np
from scipy import stats

rolnR = 0.9

def lognstats(mu, sigma, inputtype=None):
    if inputtype is None:    # mean and std to logmean and logstd
        cov = sigma/mu
        logmean = np.log(mu/np.sqrt(1.+cov**2))
        logstd = np.sqrt(np.log(1.+cov**2))
        return [logmean,logstd]
    elif inputtype == 'log':
        mean = np.exp(mu+sigma**2/2.)
        std = np.sqrt( (np.exp(sigma**2)-1.)*np.exp(2*mu+sigma**2) )
        return [mean,std]


def wblstats(mu, sigma, inputtype=None):
    if inputtype is None:
        def wblfunc(x):
            y1 = x[0]*gamma(1+1./x[1]) - mu
            y2 = x[0]**2*(gamma(1+2./x[1])-gamma(1+1./x[1])**2) - sigma**2
            return [y1,y2]
        x0 = np.array([mu, 1.2/(sigma/mu)])
        rootres = opt.root(wblfunc, x0)
        [scale, c] = rootres.x
        return [scale,c]
    elif inputtupe == 'wbl':
        c = mu; scale=sigma
        mean,var = stats.weibull_min.stats(c, scale=scale)
        return [mean, np.sqrt(var)]


def ksmp_mc(nsmp, C, Sre, G, m, Na):
    nsmp = int(nsmp)
    # correlate Csmp and msmp
    msmp = m.rvs(size=nsmp)
    mmean,mvar = m.stats(); mmean = mmean[()]; mstd = np.sqrt(mvar[()])
    umsmp = (msmp-mmean)/mstd
    cmean,cvar = C.stats(); cmean = cmean[()]; cstd = np.sqrt(cvar)
    logCmean, logCstd = lognstats(cmean, cstd)
    uLogCsmp0 = stats.norm.rvs(size=nsmp)
    uLogCsmp = -np.sqrt(rolnR**2)*umsmp-np.sqrt(1-rolnR**2)*uLogCsmp0    #correlated logC
    Csmp = np.exp(logCmean+uLogCsmp*logCstd)
    # other variables
    Sresmp = Sre.rvs(size=nsmp)
    Nasmp = Na.rvs(size=nsmp)
    ksmp = Csmp*(Sresmp**msmp)*(G**msmp)*(np.pi**(msmp/2.))*Nasmp
    return ksmp


def aismp_mc(nsmp, life, a0, C, Sre, G, m, Na):
    nsmp = int(nsmp)
    # correlate Csmp and msmp
    msmp = m.rvs(size=nsmp)
    mparam = m.stats(); mmean = mparam[0][()]; mstd = np.sqrt(mparam[1])
    umsmp = (msmp-mmean)/mstd
    cmean,cvar = C.stats(); cmean = cmean[()]; cstd = np.sqrt(cvar)
    logCmean, logCstd = lognstats(cmean, cstd)
    uLogCsmp0 = stats.norm.rvs(size=nsmp)
    uLogCsmp = -np.sqrt(rolnR**2)*umsmp-np.sqrt(1-rolnR**2)*uLogCsmp0    #correlated logC
    Csmp = np.exp(logCmean+uLogCsmp*logCstd)
    # other variables
    a0smp = a0.rvs(size=nsmp)
    Sresmp = Sre.rvs(size=nsmp)
    Nasmp = Na.rvs(size=nsmp)
    ksmp = Csmp*(Sresmp**msmp)*(G**msmp)*(np.pi**(msmp/2.))*Nasmp
    # if msmp==2
    aismp = a0smp*np.exp(ksmp*life)
    # if msmp!=2
    tmp = 1.0-msmp/2.
    indx = msmp!=2
    aismp[indx] = 1e3*( ksmp[indx]*life*tmp[indx]+(a0smp[indx]*1e-3)**tmp[indx] )**(1./tmp[indx])
    return aismp


def msr2k(rvnames, rvs, trunclb, truncub, G):
    # robustnes
    klb = trunclb[0]; kub=truncub[0];
    # reliability
    corr = np.eye(len(rvnames))
    probdata = ProbData(names=rvnames, rvs=rvs, corr=corr, nataf=False)
    analysisopt = AnalysisOpt(gradflag='DDM', recordu=False, recordx=False,
            flagsens=False, verbose=False)
    # limit state 1
    def gf1(x, param=None):
        m, C, Sre, Na = x
        K = C*(Sre**m)*(G**m)*(np.pi**(m/2.))*Na
        return K-kub
    def dgdq1(x, param=None):
        m, C, Sre, Na = x
        Srem = Sre**m; Gm = G**m; pim2 = np.pi**(m/2.)
        dgdm = C*np.log(Sre)*Srem*Gm*pim2*Na+C*Srem*np.log(G)*Gm*pim2*Na+\
               C*Srem*Gm*np.log(np.pi)*pim2*0.5*Na
        dgdC = Srem*Gm*pim2*Na
        dgdSre = C*m*(Sre**(m-1.))*Gm*pim2*Na
        dgdNa = C*Srem*Gm*pim2
        return [dgdm, dgdC, dgdSre, dgdNa]
    gfunc1 = Gfunc(gf1, dgdq1)
    formBeta1 = CompReliab(probdata, gfunc1, analysisopt)

    # limit state 2
    def gf2(x, param=None):
        m, C, Sre, Na = x
        K = C*(Sre**m)*(G**m)*(np.pi**(m/2))*Na
        return klb-K
    def dgdq2(x, param=None):
        m, C, Sre, Na = x
        Srem = Sre**m; Gm = G**m; pim2 = np.pi**(m/2)
        dgdm = C*np.log(Sre)*Srem*Gm*pim2*Na+C*Srem*np.log(G)*Gm*pim2*Na+\
               C*Srem*Gm*np.log(np.pi)*pim2*0.5*Na
        dgdC = Srem*Gm*pim2*Na
        dgdSre = C*m*(Sre**(m-1.))*Gm*pim2*Na
        dgdNa = C*Srem*Gm*pim2
        return [-dgdm, -dgdC, -dgdSre, -dgdNa]
    gfunc2 = Gfunc(gf2, dgdq2)
    formBeta2 = CompReliab(probdata, gfunc2, analysisopt)

    # system reliability
    try:
        if np.isneginf(klb):
            formresults = formBeta1.form_result()
            pf = formresults.pf1
        elif np.isposinf(kub):
            formresults = formBeta2.form_result()
            pf = formresults.pf1
        else:
            sysBeta = SysReliab([formBeta1, formBeta2], [2])
            sysformres = sysBeta.mvn_msr(sysBeta.syscorr)
            pf = sysformres.pf
        # formresults = formBeta2.form_result()
        # pf = formresults.pf1
    except np.linalg.LinAlgError:
        pf = 0.
    return pf


def mc2k(rvnames, rvs, bins, G, nsmp=1e6):
    nsmp = int(nsmp)
    # robustnes
    m, C, Sre, Na = rvs
    # correlate Csmp and msmp
    msmp = m.rvs(size=nsmp)
    mmean,mvar = m.stats(); mmean = mmean[()]; mstd = np.sqrt(mvar[()])
    umsmp = (msmp-mmean)/mstd
    cmean,cvar = C.stats(); cmean = cmean[()]; cstd = np.sqrt(cvar)
    logCmean, logCstd = lognstats(cmean, cstd)
    uLogCsmp0 = stats.norm.rvs(size=nsmp)
    uLogCsmp = -np.sqrt(rolnR**2)*umsmp-np.sqrt(1-rolnR**2)*uLogCsmp0    #correlated logC
    Csmp = np.exp(logCmean+uLogCsmp*logCstd)
    # other variables
    Sresmp = Sre.rvs(size=nsmp)
    Nasmp = Na.rvs(size=nsmp)
    ksmp = Csmp*(Sresmp**msmp)*(G**msmp)*(np.pi**(msmp/2.))*Nasmp
    ksmp = ksmp[~np.isnan(ksmp)]
    binnum, bins = np.histogram(ksmp, bins)
    binnum[binnum==0] = 0.1
    probs = binnum/np.sum(binnum,dtype=float)
    return probs


def mc2ai(rvnames, rvs, bins, nsmp=1e6):
    nsmp = int(nsmp)
    # parameters
    Ap, K, M = rvs
    # crude mc
    apsmp = Ap.rvs(size=nsmp)
    ksmp = K.rvs(size=nsmp)
    msmp = M.rvs(size=nsmp)
    # life = 1
    # # if msmp==2
    # aismp = apsmp*np.exp(ksmp*life)
    # # if msmp!=2
    # tmp = 1.0-msmp/2.
    # indx = msmp!=2
    # aismp[indx] = 1e3*( ksmp[indx]*life*tmp[indx]+(apsmp[indx]*1e-3)**tmp[indx] )**(1./tmp[indx])
    aismp = apsmp+1e3*ksmp*(apsmp*1e-3)**(msmp/2.)
    # aismp[np.isposinf(aismp)] = np.nanmax(aismp)
    # bins[np.isneginf(bins)]=np.nanmin(aismp)
    # bins[np.isposinf(bins)]=np.nanmax(aismp)
    binnum, bins = np.histogram(aismp, bins)
    binnum[binnum==0] = 0.1
    probs = binnum/np.sum(binnum,dtype=float)
    return probs, {'aismp':aismp, 'apsmp':apsmp, 'ksmp':ksmp, 'msmp':msmp}
