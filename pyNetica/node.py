from scipy import stats
from scipy.special import erf
from scipy.misc import comb
from scipy.optimize import minimize

import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore', r'divide by zero encountered in log')

_LMD_DEFAULT = 0.05
_continuous_dist = ['continuous', 'normal', 'lognormal', 'uniform']

class Node(object):
    def __init__(self, name, **kwargs):
        self.name = name
        try:
            self.parents = kwargs['parents']
        except KeyError:
            print "keyword parameter \'parents\' must be provided"
            sys.exit(1)
        try:
            self.rvname = kwargs['rvname']
        except KeyError:
            print "keyword parameter \'rvname\' must be provided"
            sys.exit(1)
        try:
            self.rv = kwargs['rv']
        except KeyError:
            self.rv = None
        self.ptr = None
        self.net = None
        self.cpt = None
        self.statenames = None
        self.bins = None


    def add_to_net(self, net):
        """
        add to network net
        """
        if np.any(self.cpt==-1):
            print "CPT of node {} has not been fully assigned".format(self.name)
            sys.exit(1)
        self.net = net
        self.ptr = net.ntc.newnode(self.name, self.cpt.shape[1], net.net)
        # set node names
        stringnames = np.array2string(np.array(self.statenames), separator=',')[1:-1]
        stringnames = stringnames.replace('\'', '')
        self.net.ntc.setnodestatenames(self.ptr, stringnames)


    def define(self):
        """
        define nodes in the network
        """
        if self.net is None:
            print "node must be added to the network first"
            sys.exit(1)
        # set node probs
        if self.parents is None:
            pstates = np.empty((1,), dtype='int32')
            probs = self.cpt[-1].astype('float32')
            self.net.ntc.setnodeprobs (self.ptr, pstates, probs)
        else:
            # from icpt to label
            nparent = np.array(self.parents).size
            npstate = np.empty((nparent,), dtype=int); npstate.fill(-1)
            for iparent, parent in enumerate(self.parents):
                npstate[iparent] = parent.cpt.shape[1]
            # label list
            import itertools
            command = "labels = list(itertools.product(np.arange(npstate[0])"
            for ipstate in npstate[1:]:
                command += ", np.arange({})".format(ipstate)
            command += "))"
            exec command
            if len(labels) != self.cpt.shape[0]:
                print "algorithm error"
                sys.exit(1)
            for i, label in enumerate(labels):
                pstates = np.array(label).astype('int32')
                probs = self.cpt[i,:].astype('float32')
                self.net.ntc.setnodeprobs(self.ptr, pstates, probs)

    def nstates(self):
        return self.cpt.shape[1]


    def discretize(self, lb, ub, num, infinity=None, bins=None):
        if bins is None:
            if infinity is None:
                bins = np.linspace(lb, ub, num=num+1)
            elif infinity == '+' or infinity == np.inf:
                bins = np.linspace(lb, ub, num=num)
                bins = np.append(bins, np.inf)
            elif infinity == '-' or infinity == -np.inf:
                bins = np.linspace(lb, ub, num=num)
                bins = np.insert(bins, 0, -np.inf)
            elif infinity == '+-':
                bins = np.linspace(lb, ub, num=num-1)
                bins = np.insert(bins, 0, -np.inf)
                bins = np.append(bins, np.inf)
        else:
            if (infinity == '+' or infinity == np.inf) and (bins[-1] != np.inf):
                bins = np.append(bins, np.inf)
            elif (infinity == '-' or infinity == -np.inf) and (bins[0] != -np.inf):
                bins = np.insert(bins, 0, -np.inf)
            elif infinity == '+-' and (bins[0]!=-np.inf) and (bins[-1]!=np.inf):
                bins = np.insert(bins, 0, -np.inf)
                bins = np.append(bins, np.inf)
        self.bins = bins
        return (0.5*(bins[:-1]+bins[1:])).astype('string')


    def truncate_rv(self, ntrunct, lmd=_LMD_DEFAULT):
        """ntrunct: the number of truncated rv, start from 0, cannot be negative"""
        # arg check
        if ntrunct<0:
            print "negative ntrunct is not supported yet"
            sys.exit(1)
        notdiscretized = self.bins is None or \
                         (self.bins[ntrunct] == -np.inf and self.bins[ntrunct+1] == np.inf)
        if self.rvname in _continuous_dist and notdiscretized:
            print "node {} has not been discretized yet".format(self.name)
            sys.exit(1)

        # for continuous rv:
        bins = self.bins
        if self.rvname in _continuous_dist:
            if self.rv is None:
                if self.bins[ntrunct] == -np.inf:
                    redge = bins[ntrunct+1]
                    trv = NegExpon(loc=redge, scale=1./lmd)
                elif self.bins[ntrunct+1] == np.inf:
                    ledge = bins[ntrunct]
                    trv = stats.expon(loc=ledge, scale=1./lmd)
                else:
                    ledge = bins[ntrunct]
                    redge = bins[ntrunct+1]
                    trv = stats.uniform(loc=ledge, scale = redge-ledge)
            else:
                ledge = bins[ntrunct]
                redge = bins[ntrunct+1]
                trv = TrancatedRv(self.rv, ledge, redge, self.rvname)

        # for discrete rv
        else:
            trv = self.statenames[ntrunct]

        return trv


    def assign_cpt(self, cpt, label=None, statenames=None):
        """ when label is given, cpt is a 1d-array; otherwise, it must be a 2d-array"""
        statenum = cpt.shape[-1]
        # set node names
        if statenames is None:
            self.statenames = np.arange(statenum)+1
        elif np.size(statenames) != statenum:
            print "for node {}, size of statenames must be the same as statenum".format(self.name)
            sys.exit(1)
        else:
            self.statenames = statenames
        if label is None or self.parents is None:
            self.cpt = cpt
        else:
            nparent = np.array(self.parents).size
            if label.size != nparent:
                print "label must be of same size of self.parents"
                sys.exit(1)
            kweight = np.empty(label.shape); kweight.fill(-1.0)
            npstate = np.empty(label.shape, dtype=int); npstate.fill(-1)
            #label to cpt index
            for iparent, parent in enumerate(self.parents):
                try:
                    npstate[iparent] = parent.cpt.shape[1]
                except IndexError:
                    print "parent {} of node {} must be assigned to a cpt".format(parent.name, self.name)
                    sys.exit(1)
                kweight[iparent] = npstate[iparent]**(nparent-iparent-1)
            #if never assigned, initialize cpt
            if self.cpt is None:
                self.cpt = np.empty((np.prod(npstate),statenum));self.cpt.fill(-1.0)
            icpt = np.dot(label, kweight).astype(int)
            self.cpt[icpt,:] = cpt


    def node_stats(self, probs, states=None, lmd=_LMD_DEFAULT, moments='mv'):
        """ only after belief updating
            states: state values in float
            probs: prior or posterior probs
            lmd: lmd for exponential distribution, must be the same as in assign_cpt
            moments: same as rv_frozen objects, only 'm' and 'v' are supported
        """
        if states is None:
            rvs = [self.truncate_rv(i, lmd=lmd) for i in xrange(self.nstates())]
            mm1 = [rv.moment(1) for rv in rvs]
            mm2 = [rv.moment(2) for rv in rvs]
        else:
            mm1 = states
            mm2 = states**2
        sts = []
        for moment in moments:
            if moment == 'm':
                sts.append(np.dot(mm1, probs))
            elif moment == 'v':
                m1 = np.dot(mm1, probs)
                m2 = np.dot(mm2, probs)
                sts.append(m2-m1**2)
            else:
                print "moments not supported"
                sys.exit(1)
        if len(sts) == 1:
            return sts[0]
        else:
            return tuple(sts)


    def node_cdf(self, xarray, probs, lmd = _LMD_DEFAULT):
        """ for continuous nodes only"""
        iterable = isinstance(xarray, (np.ndarray, list, tuple))
        if not iterable:
            xarray = [xarray]
        cdfs = np.zeros(np.array(xarray).shape)
        for i, x in enumerate(xarray):
            rvindx = np.searchsorted(self.bins, x)-1
            if rvindx<0:
                rvindx = 0
            rv = self.truncate_rv(rvindx, lmd)
            cdfs[i] = np.sum(probs[:rvindx])+rv.cdf(x)*probs[rvindx]
        if iterable:
            return cdfs
        else:
            return cdfs[0]


    def node_pdf(self, xarray, probs, lmd = _LMD_DEFAULT):
        """ for continuous nodes only"""
        iterable = isinstance(xarray, (np.ndarray, list, tuple))
        if not iterable:
            xarray = [xarray]
        pdfs = np.zeros(np.array(xarray).shape)
        for i, x in enumerate(xarray):
            rvindx = np.searchsorted(self.bins, x)-1
            if rvindx<0:
                rvindx = 0
            rv = self.truncate_rv(rvindx, lmd)
            pdfs[i] = rv.pdf(x)*probs[rvindx]
        if iterable:
            return pdfs
        else:
            return pdfs[0]


    def fit_stats(self, rvname, probs, lmd=_LMD_DEFAULT, x0=None):
        """ for intermediate nodes (nodes with parents) only"""
        if np.isinf(self.bins[-1]) and np.isinf(self.bins[0]):
            loc = self.bins[1:-1]
        elif np.isinf(self.bins[-1]):
            loc = self.bins[:-1]
        elif np.inf(self.bins[-1]):
            loc = self.bins[1:]
        else:
            loc = self.bins
        cdfdata = self.node_cdf(loc, probs, lmd=lmd)
        if x0 is None:
            x0 = self.node_stats(probs, lmd=lmd)
        if rvname.lower() == 'lognormal':
            def objfunc(x, loc, cdfdata):
                m = x[0]
                v = x[1]
                mu = np.log(m/np.sqrt(1+v/m**2))
                sgm = np.sqrt(np.log(1+v/m**2))
                return np.linalg.norm(stats.lognorm.cdf(loc, sgm, scale=np.exp(mu))-cdfdata)
            optres = minimize(objfunc, x0, args = (loc,cdfdata), bounds=([0., np.inf], [0., np.inf]))
        else:
            print "unknown distribution name to fit cdf data to"
            sys.exit(1)
        return optres.x, optres


class TrancatedRv(object):
    def __init__(self, rv, lb, ub, rvname=None):
        self.rv = rv
        self.lb = lb
        self.ub = ub
        self.rvname = rvname

    def pdf(self, x):
        rv = self.rv
        ub = self.ub
        lb = self.lb
        rvpdf = rv.pdf(x) / (rv.cdf(ub)-rv.cdf(lb))
        try:
            rvpdf[np.logical_or(x<lb, x>ub)] = 0.
        except TypeError:
            if x<lb or x>ub:
                rvpdf = 0.
        return rvpdf

    def cdf(self, x):
        rv = self.rv
        ub = self.ub
        lb = self.lb
        rvcdf = (rv.cdf(x)-rv.cdf(lb)) / (rv.cdf(ub)-rv.cdf(lb))
        try:
            rvcdf[x<lb] = 0.
            rvcdf[x>ub] = 1.
        except TypeError:
            if x<lb:
                rvcdf = 0.
            elif x>ub:
                rvcdf = 1.
        return rvcdf

    def ppf(self, x):
        rv = self.rv
        ub = self.ub
        lb = self.lb
        rv0cdf = x*(rv.cdf(ub)-rv.cdf(lb))+rv.cdf(lb)
        rvppf = rv.ppf(rv0cdf)
        return rvppf

    def rvs(self, size=None):
        rvcdf = np.random.rand(size)
        rvsmp = self.ppf(rvcdf)
        return rvsmp

    def moment(self, order, nsmp=1e4):
        l = self.lb; u = self.ub
        rvname = self.rvname
        if rvname.lower() == "normal" and order<=3:
            dnmntr = self.rv.cdf(u) - self.rv.cdf(l)
            mu,var = self.rv.stats()
            sgm = np.sqrt(var)
            k0 = sgm/np.sqrt(2*np.pi)
            k1 = (u-mu)/sgm/np.sqrt(2)
            k2 = (l-mu)/sgm/np.sqrt(2)
            #mm0 = 0.5*(erf(k1) - erf(k2))
            mm0 = stats.norm.cdf(np.sqrt(2)*k1)-stats.norm.cdf(np.sqrt(2)*k2)
            if order == 1:
                mm = k0*(np.exp(-k2**2)-np.exp(-k1**2))+mu*mm0
            elif order == 2:
                mm = k0*((l+u)*np.exp(-k2**2)-(u+mu)*np.exp(-k1**2))+(mu**2+sgm**2)*mm0
            elif order == 3:
                mm = k0*((2*sgm**2+mu**2+l*mu+l**2)*np.exp(-k2**2) -
                         (2*sgm**2+mu**2+u*mu+u**2)*np.exp(-k1**2))+mu*(mu**2+3*sgm**2)*mm0
            mm = mm/dnmntr
        elif rvname.lower() == "lognormal":
            dnmntr = self.rv.cdf(u) - self.rv.cdf(l)
            m,v = self.rv.stats()
            mu = np.log(m/np.sqrt(1+v/m**2))
            sgm = np.sqrt(np.log(1+v/m**2))
            N = order
            k1 = (np.log(u)-mu)/sgm/np.sqrt(2) - N*sgm/np.sqrt(2)
            k2 = (np.log(l)-mu)/sgm/np.sqrt(2) - N*sgm/np.sqrt(2)
            d = 2*(stats.norm.cdf(np.sqrt(2)*k1)-stats.norm.cdf(np.sqrt(2)*k2))
            mm = 0.5*np.exp(N*mu+N**2*sgm**2/2)*d
            mm = mm/dnmntr
        else:
            nsmp = int(nsmp)
            smp = self.rvs(size=nsmp)
            mm = np.mean( smp**(int(order)) )
        return mm

    def stats(self, moments='mv', nsmp=1e4):
        sts = []
        useCloseMoments = self.rvname.lower() == "normal" or\
                          self.rvname.lower() == "lognormal"
        if useCloseMoments:
            for moment in moments:
                if moment == 'm':
                    sts.append(np.array(self.moment(1)))
                elif moment == 'v':
                    ex1 = self.moment(1)
                    ex2 = self.moment(2)
                    sts.append(np.array(ex2-ex1**2))
        else:
            nsmp = int(nsmp)
            smp = self.rvs(size=nsmp)
            for moment in moments:
                if moment == 'm':
                    sts.append(np.array(np.mean(smp)))
                elif moment == 'v':
                    sts.append(np.array(np.var(smp)))

        if len(sts) == 1:
            return sts[0]
        else:
            return tuple(sts)

    def mean(self):
        return self.stats(moments='m')


class NegExpon(object):
    def __init__(self, loc=0, scale=1.):
        self.rv = stats.expon(loc=loc, scale=scale)
        self.loc = loc
        self.scale = scale

    def pdf(self, x):
        y = 2*self.loc - x
        return self.rv.pdf(y)

    def cdf(self, x):
        y = 2*self.loc - x
        return self.rv.cdf(y)

    def ppf(self, x):
        y = self.rv.ppf(x)
        return 2*self.loc - y

    def rvs(self, size=None):
        y = self.rv.rvs(size)
        return 2*self.loc - y

    def moment(self, order):
        if order>2:
            print "moments of higher order than 2 is not supported"
            sys.exit(1)
        if order == 1:
            return 2*self.loc - self.rv.moment(1)
        elif order == 2:
            mm1 = self.rv.moment(1)
            mm2 = self.rv.moment(2)
            k = self.loc
            return mm2-4.*k*mm1 + 4.*k**2

    def stats(self, moments='mv'):
        keys = moments
        ystats = self.rv.stats(moments)
        ydict = dict(zip(keys, ystats))
        sts = []
        for moment in moments:
            if moment == 'm':
                sts.append(2*self.loc - ydict['m'])
            elif moment == 'v':
                sts.append(ydict['v'])
            elif moment == 's':
                sts.append(2*self.loc - ydict['s'])
            elif moment == 'k':
                sts.append(ydict['k'])
        if len(sts) == 1:
            return sts[0]
        else:
            return tuple(sts)

    def mean(self):
        return 2*self.loc-self.rv.mean()
