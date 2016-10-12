import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from soliman2014_funcs import lognstats, wblstats, rolnR

# check the error brought by explicit integration

acrit = 50.


def aismp_mc(nsmp, life, a0smp, Csmp, Sresmp, G, msmp, Nasmp, implicit=False):
    nsmp = int(nsmp)
    ksmp = Csmp*(Sresmp**msmp)*(G**msmp)*(np.pi**(msmp/2.))*Nasmp
    if implicit is True:
        # if msmp==2
        aismp = a0smp*np.exp(ksmp*life)
        # if msmp!=2
        tmp = 1.0-msmp/2.
        indx = msmp!=2.
        # aismp[indx] = 1e3*( ksmp[indx]*life*tmp[indx]+(a0smp[indx]*1e-3)**tmp[indx] )**(1./tmp[indx])
        diff = ksmp[indx]*life*tmp[indx]+(a0smp[indx])**tmp[indx]
        aismp[indx&(diff>=0)] = diff[indx&(diff>=0)]**(1./tmp[indx&(diff>=0)])
        aismp[indx&(diff<0)] = acrit + 1e-3
        aismp[aismp>acrit] = acrit + 1e-3
    else:
        # aismp = a0smp+1e3*ksmp*(a0smp*1e-3)**(msmp/2.)
        aismp = a0smp+ksmp*(a0smp)**(msmp/2.)
        aismp[aismp>acrit] = acrit + 1e-3
    return aismp


if __name__ == '__main__':
    # crude MC for no evidence
    nsmp = int(1e6)
    G = 1.12
    lmd = 0.122; beta = -0.305    # w.r.t. mm
    sigmae = 0.2    # mm
    life=20; lifearray = np.arange(life+1.)
    lifearray1 = np.arange(0, life+1., 5.)
    # random variables
    a0mean,a0std = 0.5, 0.5*0.1
    rv_a0 = stats.norm(a0mean,a0std)
    mmean,mstd = 3.0, 3.0*0.1
    rv_m = stats.norm(mmean, mstd)
    [logCmean, logCstd] = lognstats(2.3e-12, 0.3*2.3e-12)
    rv_C = stats.lognorm(logCstd, scale=np.exp(logCmean))
    [wblscale, wblc] = wblstats(22.5, 0.1*22.5)
    rv_Sre = stats.weibull_min(wblc, scale=wblscale)
    [logNamean, logNastd] = lognstats(1e6, 0.1*1e6)
    rv_Na = stats.lognorm(logNastd, scale=np.exp(logNamean))
    [logNamean1, logNastd1] = lognstats(5e6, 0.1*5e6)
    rv_Na1 = stats.lognorm(logNastd1, scale=np.exp(logNamean1))
    # crude MC
    # correlate Csmp and msmp
    msmp = rv_m.rvs(size=nsmp)
    umsmp = (msmp-mmean)/mstd
    uLogCsmp0 = stats.norm.rvs(size=nsmp)
    uLogCsmp = -np.sqrt(rolnR**2)*umsmp-np.sqrt(1-rolnR**2)*uLogCsmp0    #correlated logC
    Csmp = np.exp(logCmean+uLogCsmp*logCstd)
    # other variables
    a0smp = rv_a0.rvs(size=nsmp)
    Sresmp = rv_Sre.rvs(size=nsmp)
    Nasmp = rv_Na.rvs(size=nsmp)
    Na1smp = rv_Na1.rvs(size=nsmp)
    # ksmp = Csmp*(Sresmp**msmp)*(G**msmp)*(np.pi**(msmp/2.))*Nasmp
    # time series
    asmparrayExp = [a0smp]
    asmparrayImp = [a0smp]
    for t in lifearray[1:]:
        asmpExp = aismp_mc(1e6, t, asmparrayExp[-1], Csmp, Sresmp, G, msmp, Nasmp, implicit=False)
        asmpImp = aismp_mc(1e6, 1., asmparrayImp[-1], Csmp, Sresmp, G, msmp, Nasmp, implicit=True)
        asmparrayExp.append(asmpExp)
        asmparrayImp.append(asmpImp)
    ameanExp = np.nanmean(asmparrayExp, axis=1)
    ameanImp = np.nanmean(asmparrayImp, axis=1)
    astdExp = np.nanstd(asmparrayExp, axis=1)
    astdImp = np.nanstd(asmparrayImp, axis=1)
    asmparrayImp1 = [a0smp]
    for t in lifearray1[1:]:
        asmpImp1 = aismp_mc(1e6, 1., asmparrayImp1[-1], Csmp, Sresmp, G, msmp, Na1smp, implicit=True)
        asmparrayImp1.append(asmpImp1)
    ameanImp1 = np.nanmean(asmparrayImp1, axis=1)
    astdImp1 = np.nanstd(asmparrayImp1, axis=1)


    plt.figure()
    plt.plot(lifearray, ameanExp)
    plt.plot(lifearray, ameanImp)
    plt.plot(lifearray1, ameanImp1)
    plt.figure()
    plt.plot(lifearray, astdExp)
    plt.plot(lifearray, astdImp)
    plt.plot(lifearray1, astdImp1)
