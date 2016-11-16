import pymc
import numpy as np
from scipy import stats

from soliman2014_funcs import lognstats, wblstats, ksmp_mc, rolnR

if __name__ == '__main__':
    # crude MC for no evidence
    nsmp = int(1e6)
    G = 1.12
    lmd = 0.122; beta = -0.305    # w.r.t. mm
    sigmae = 0.2    # mm
    acrit = 30.
    life=5; lifearray = np.arange(life)+1.
    # random variables
    a0mean,a0std = 0.5, 0.5*0.1
    rv_a0 = stats.norm(a0mean,a0std)
    mmean,mstd = 3.0, 3.0*0.05
    rv_m = stats.norm(mmean, mstd)
    [logCmean, logCstd] = lognstats(2.3e-12, 0.3*2.3e-12)
    # [logCmean, logCstd] = lognstats(4.5e-13, 0.3*4.5e-13)
    rv_C = stats.lognorm(logCstd, scale=np.exp(logCmean))
    [wblscale, wblc] = wblstats(22.5, 0.1*22.5)
    rv_Sre = stats.weibull_min(wblc, scale=wblscale)
    [logNamean, logNastd] = lognstats(2e6, 0.1*2e6)
    rv_Na = stats.lognorm(logNastd, scale=np.exp(logNamean))
    # crude MC
    # correlate Csmp and msmp
    msmp = rv_m.rvs(size=nsmp)
    umsmp = (msmp-mmean)/mstd
    uLogCsmp0 = stats.norm.rvs(size=nsmp)
    uLogCsmp = -np.sqrt(rolnR**2)*umsmp-np.sqrt(1-rolnR**2)*uLogCsmp0    #correlated logC
    Csmp = np.exp(logCmean+uLogCsmp*logCstd)
    # other variables
    Sresmp = rv_Sre.rvs(size=nsmp)
    Nasmp = rv_Na.rvs(size=nsmp)
    ksmp = Csmp*(Sresmp**msmp)*(G**msmp)*(np.pi**(msmp/2.))*Nasmp
    # time series
    asmparray = [rv_a0.rvs(size=nsmp)]
    for t in lifearray:
        # asmp = asmparray[-1] + 1e3*ksmp*(asmparray[-1]*1e-3)**(msmp/2.)
        # if msmp==2
        apsmp = asmparray[-1]
        aismp = apsmp*np.exp(ksmp*1.)
        # if msmp!=2
        tmp = 1.0-msmp/2.
        indx = msmp!=2
        diff = ksmp[indx]*1.*tmp[indx]+(apsmp[indx])**tmp[indx]
        aismp[indx&(diff>=0)] = diff[indx&(diff>=0)]**(1./tmp[indx&(diff>=0)])
        aismp[indx&(diff<0)] = acrit + 1e-3
        aismp[aismp>acrit] = acrit + 1e-3
        asmparray.append(aismp)
    # failure probability
    pfarray = []
    for asmp in asmparray:
        pf = np.sum(asmp>acrit,dtype=float)/nsmp
        pfarray.append(pf)

    # MCMC using pymc
    def make_pymc_model():
        # data
        madata = np.empty(life, dtype=object)
        madata[1] = 1.0
        madata[3] = 1.0

        # parameters
        cs = pymc.Normal('CS', mu=0., tau=1., plot=False)
        m = pymc.Normal('M', mu=mmean, tau=1./mstd**2, plot=False)
        Sre = pymc.Weibull('Sre', alpha=wblc, beta=wblscale, plot=False)
        Na = pymc.Lognormal('Na', mu=logNamean, tau=1./logNastd**2, plot=False)
        a0 = pymc.Normal('A0', mu=a0mean, tau=1./a0std**2, plot=True)

        #transformed parameters
        @pymc.deterministic(name='C', plot=False)
        def C(cs=cs, m=m):
            um = (m-mmean)/mstd
            uLogC = -np.sqrt(rolnR**2)*um-np.sqrt(1-rolnR**2)*cs
            Cout = np.exp(logCmean+uLogC*logCstd)
            return Cout

        @pymc.deterministic(name='Kappa', plot=False)
        def kappa(C=C, Sre=Sre, m=m, Na=Na):
            kout = C*(Sre**m)*(G**m)*(np.pi**(m/2.))*Na
            return kout
        aarray = np.empty(life+1, dtype=object)
        aarray[0] = a0
        for i in np.arange(1,life+1):
            @pymc.deterministic(name='A{}'.format(i), plot=True)
            def ai(kappa=kappa, m=m, ap=aarray[i-1]):
                # aiout = ap+1e3*kappa*(ap*1e-3)**(m/2.)
                if m==2:
                    aiout = ap*np.exp(kappa*1.)
                else:
                    tmp = 1.0-m/2.
                    if ap<=0: ap=1e-12
                    diff = kappa*1.*tmp+ap**tmp
                    if diff>=0:
                        aiout = diff**(1./tmp)
                    else:
                        aiout = acrit + 1e-3
                    if aiout>acrit:
                        aiout = acrit + 1e-3
                return aiout
            aarray[i] = ai


        # observable variable
        Marray = np.empty(2, dtype=object)
        for mindx,i in enumerate([2, 4]):
            mivalue = madata[i-1]
            if i == 2:
                mledge = 0.50669
                mredge = 0.58725
            elif i==4:
                mledge = 3.47208
                mredge = 4.13522
            obsvalue = True
            @pymc.stochastic(name='M{}'.format(i), plot=False, dtype=int, observed=obsvalue)
            def Mi(value=mivalue, ai=aarray[i]):
                def logp(value, ai):
                    pod = 1.-stats.norm.cdf((np.log(ai)-lmd)/beta)
                    pinrange = stats.norm.cdf(mredge, loc=ai, scale=sigmae)-\
                            stats.norm.cdf(mledge, loc=ai, scale=sigmae)
                    if value == 0:
                        p = (1.-pod)+pod*(1.-pinrange)
                        return np.log(p)
                    else:
                        return np.log(pod*pinrange)
                def random(ai):
                    pod = 1.-stats.norm.cdf((np.log(ai)-lmd)/beta)
                    pinrange = stats.norm.cdf(mredge, loc=ai, scale=sigmae)-\
                            stats.norm.cdf(mledge, loc=ai, scale=sigmae)
                    if np.random.rand()<=pod:
                        return stats.bernoulli.rvs(p=pinrange,size=1)
                    else:
                        return 0
            Marray[mindx] = Mi

        return locals()

    M = pymc.MCMC(make_pymc_model(), db='pickle', dbname='soliman_pymc.pickle')
    # graph = pymc.graph.graph(M)
    # graph.write_png('soliman2014_pymc.png')
    # import os
    # os.system('eog soliman2014_pymc.png')

    nchain=4; niter=int(1e6); nburn=niter/2; nthin=2
    for i in range(nchain):
        M.sample(iter=niter, burn=nburn, thin=nthin)
    pymc.Matplot.plot(M)
    M.db.close()
