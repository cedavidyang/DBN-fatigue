import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from soliman2014_funcs import lognstats, wblstats, rolnR

# check the error brought by explicit integration


def aismp_mc(nsmp, life, a0smp, Csmp, Sresmp, G, msmp, Nasmp, implicit=False):
    nsmp = int(nsmp)
    ksmp = Csmp*(Sresmp**msmp)*(G**msmp)*(np.pi**(msmp/2.))*Nasmp
    if implicit is True:
        # if msmp==2
        aismp = a0smp*np.exp(ksmp*life)
        # if msmp!=2
        tmp = 1.0-msmp/2.
        indx = msmp!=2
        aismp[indx] = 1e3*( ksmp[indx]*life*tmp[indx]+(a0smp[indx]*1e-3)**tmp[indx] )**(1./tmp[indx])
    else:
        aismp = a0smp+1e3*ksmp*(a0smp*1e-3)**(msmp/2.)
    return aismp


if __name__ == '__main__':
    # crude MC for no evidence
    nsmp = int(1e6)
    G = 1.12
    lmd = 0.122; beta = -0.305    # w.r.t. mm
    sigmae = 0.2    # mm
    life=5; lifearray = np.arange(life+1.)
    # random variables
    a0mean,a0std = 0.5, 0.5*0.1
    rv_a0 = stats.norm(a0mean,a0std)
    mmean,mstd = 3.0, 3.0*0.1
    rv_m = stats.norm(mmean, mstd)
    [logCmean, logCstd] = lognstats(2.3e-12, 0.3*2.3e-12)
    rv_C = stats.lognorm(logCstd, scale=np.exp(logCmean))
    [wblscale, wblc] = wblstats(22.5, 0.1*22.5)
    rv_Sre = stats.weibull_min(wblc, scale=wblscale)
    [logNamean, logNastd] = lognstats(5e6, 0.1*5e6)
    rv_Na = stats.lognorm(logNastd, scale=np.exp(logNamean))
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
    # ksmp = Csmp*(Sresmp**msmp)*(G**msmp)*(np.pi**(msmp/2.))*Nasmp
    # time series
    asmparrayExp = [a0smp]
    asmparrayImp = [a0smp]
    for t in lifearray[1:]:
        asmpExp = aismp_mc(1e6, t, asmparrayExp[-1], Csmp, Sresmp, G, msmp, Nasmp, implicit=False)
        asmpImp = aismp_mc(1e6, 1., asmparrayImp[-1], Csmp, Sresmp, G, msmp, Nasmp, implicit=True)
        asmparrayExp.append(asmpExp)
        asmparrayImp.append(asmpImp)
    ameanExp = np.mean(asmparrayExp, axis=1)
    ameanImp = np.mean(asmparrayImp, axis=1)
    astdExp = np.std(asmparrayExp, axis=1)
    astdImp = np.std(asmparrayImp, axis=1)
    plt.figure()
    plt.plot(lifearray, ameanExp)
    plt.plot(lifearray, ameanImp)
    plt.figure()
    plt.plot(lifearray, astdExp)
    plt.plot(lifearray, astdImp)
