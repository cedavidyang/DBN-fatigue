import pystan
import pymc

import numpy as np
from scipy import stats

# crude MC for no evidence
nsmp = int(1e6)
rolnR = 0.3
covar = rolnR**2*np.ones((5,5))
for i in range(5):
    covar[i,i] = 1.0
logmean = np.log(150./np.sqrt(1+0.2**2))
logstd = np.sqrt(np.log(1+0.2**2))
# z15 = stats.multivariate_normal.rvs(mean=None, cov=covar, size=nsmp)
ur = stats.norm.rvs(size=nsmp)
ur = np.array([ur, ur, ur, ur, ur]).T
u15 = stats.multivariate_normal.rvs(cov=np.eye(5), size=nsmp)
z15 = np.sqrt(1.-rolnR)*u15+np.sqrt(rolnR)*ur
r15 = np.exp(logmean+z15*logstd)
beta = (50.*0.4)/(np.pi/np.sqrt(6))
mu = 50.-np.euler_gamma*beta
h = stats.gumbel_r.rvs(loc=mu, scale=beta, size=nsmp)
theta = (60*0.2)**2/60.
k = 60./theta
v = stats.gamma.rvs(k, scale=theta,size=nsmp)
r1 = r15[:,0]
r2 = r15[:,1]
r3 = r15[:,2]
r4 = r15[:,3]
r5 = r15[:,4]
g1 = r1+r2+r4+r5-5*h
g2 = r2+2*r3+r4-5*v
g3 = r1+2*r3+2*r4+r5-5*h-5*v
gsys = np.min([g1>=0,g2>=0,g3>=0], axis=0)
pf0 = 1.-np.sum(gsys,dtype=float)/gsys.size
print 'Pf without ev (crude MC): {}'.format(pf0)

analysis_tool = input('analysis tools? 1 for stan; 2 for pymc; 3 for pymc3 ')

if analysis_tool == 1:
    # MCMC using pystan (further debugging needed)
    stan_code = """
    data {
    int<lower=0> NR;
    real logmean;
    real logstd;
    real rolnR;
    }

    parameters {
    real M4;
    real M5;
    real UR;    //common cause factor to introduce correlation
    vector<lower=0>[NR] R15;    //resistance without measurement
    real<lower=0> H;
    real V;
    }

    transformed parameters {
    vector[NR] Z15;
    vector[NR] U15;    //resistance without measurement
    real G1;
    real G2;
    real G3;
    vector[NR-2] Garray;
    real Gsys;    //system performance

    Z15 = (log(R15)-logmean)/logstd;
    U15 = (Z15-sqrt(rolnR)*UR)/sqrt(1.0-rolnR);

    G1 = R15[1] + R15[2] + R15[4] + R15[5] - 5*H;
    G2 = R15[2] + 2*R15[3] + R15[4] - 5*V;
    G3 = R15[1] + 2*R15[3] + 2*R15[4] + R15[5] - 5*H- 5*V;

    Garray[1] = G1>=0;
    Garray[2] = G2>=0;
    Garray[3] = G3>=0;
    Gsys = min(Garray);
    }

    model {

    UR ~ normal(0,1);
    for (i in 1:NR)
        U15[i] ~ normal(0,1);
    M4 ~ normal(R15[4], 15);
    M5 ~ normal(R15[5], 15);
    H ~ gumbel(40.9989,15.5939);
    V ~ gamma(25, 0.41667);

    }
    """

    # stan_data = {'NR': 5,
                # 'logmean': 4.991,
                # 'logstd': 0.19804,
                # 'rolnR': 0.3,
                # 'M4': 50,
                # 'M5': 100}
    stan_data = {'NR': 5,
                'logmean': 4.991,
                'logstd': 0.19804,
                'rolnR': 0.3}

    t = pystan.stan(model_code=stan_code, data=stan_data, iter=1000000, chains=4, n_jobs=2)
    # t = pystan.stan(model_code=stan_code, data=stan_data, iter=100, chains=4)
    smpdata = t.extract(permuted=True)
    redata = smpdata['Gsys']
    pf = 1.-np.sum(redata,dtype=float)/redata.size
    r15 = smpdata['R15']
    h = smpdata['H']
    v = smpdata['V']
    r1 = r15[:,0]
    r2 = r15[:,1]
    r3 = r15[:,2]
    r4 = r15[:,3]
    r5 = r15[:,4]
    g1 = r1+r2+r4+r5-5*h
    g2 = r2+2*r3+r4-5*v
    g3 = r1+2*r3+2*r4+r5-5*h-5*v
    gsys = np.min([g1>=0,g2>=0,g3>=0], axis=0)
    pf2 = 1.-np.sum(gsys,dtype=float)/gsys.size

elif analysis_tool == 2:
    # MCMC using pymc
    def make_pymc_model():
        # data
        m45data = np.array([150., 200.])

        # parameters
        NR = 5
        ur = pymc.Normal('UR', mu=0, tau=1., plot=False)

        u15 = np.empty(NR, dtype=object)
        for i in range(NR):
            u15[i] = pymc.Normal('u_{}'.format(i+1), mu=0, tau=1., plot=False)
        theta = (60*0.2)**2/60.
        k = 60./theta
        alpha=k; beta=1./theta
        v = pymc.Gamma('V', alpha=alpha,beta=beta, plot=False)
        beta = (50.*0.4)/(np.pi/np.sqrt(6))
        mu = 50.-np.euler_gamma*beta
        h = pymc.ScipyDistributions.Gumbel_r(name='H', loc=mu, scale=beta, plot=False)

        #transformed parameters
        r15 = np.empty(NR, dtype=object)
        for i,ui in enumerate(u15):
            @pymc.deterministic(name='r{}'.format(i+1), plot=True)
            def ri(ur=ur, ui=ui):
                zi = np.sqrt(1.-rolnR)*ui+np.sqrt(rolnR)*ur
                riout = np.exp(logmean+zi*logstd)
                return riout
            r15[i]=ri

        @pymc.deterministic(plot=False)
        def gsys(r15=r15, h=h, v=v):
            r1 = r15[0]; r2 = r15[1]; r3=r15[2]
            r4 = r15[3]; r5 = r15[4]
            g1 = r1+r2+r4+r5-5*h
            g2 = r2+2*r3+r4-5*v
            g3 = r1+2*r3+2*r4+r5-5*h-5*v
            gsysout = np.min([g1>=0,g2>=0,g3>=0], axis=0)
            return gsysout

        # observable variable
        m45 = np.empty(2, dtype=object)
        for i, midata,ri in zip(range(2), m45data, r15[-2:]):
            mi = pymc.Normal('m{}'.format(i+4), mu=ri, tau=1./15.**2, value=midata, observed=True)
            m45[i] = mi
        return locals()

    M = pymc.MCMC(make_pymc_model(), db='pickle', dbname='pymc2_straub2010ex1_m45_150_200.pickle')
    # graph = pymc.graph.graph(M)
    # graph.write_png('pymcGraph.png')
    # import os
    # os.system('eog pymcGraph.png')

    nchain=4; niter=int(1e6); nburn=niter/2; nthin=2
    for i in range(nchain):
        M.sample(iter=niter, burn=nburn, thin=nthin)
    pymc.Matplot.plot(M)
    gsyssmp = M.trace('gsys')[:]
    pf2 = 1.-np.sum(gsyssmp,dtype=float)/gsyssmp.size
    M.db.close()
