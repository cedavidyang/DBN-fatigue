import os
strepath = os.path.join(os.path.expanduser('~'), 'FERUM')
import sys
sys.path.append(strepath)

from pyStRe import *

import numpy as np
#from scipy import stats

# function from x2 and x3 to y
def form2y(rvnames, rvs, corr, bins=[0.,0.]):
    probdata = ProbData(names=rvnames, rvs=rvs, corr=corr, nataf=False)
    def gf2y(x, param=None):
        return x[1]-x[0]
    def dgf2y(x, param=None):
        dgd1 = -1.
        dgd2 = 1.
        return [dgd1, dgd2]
    analysisopt = AnalysisOpt(gradflag='DDM', recordu=False, recordx=False, flagsens=False, verbose=False)
    gfunc = Gfunc(gf2y, dgf2y)
    formBeta = CompReliab(probdata, gfunc, analysisopt)
    formresults = formBeta.form_result()
    return formresults.pf1

