from __future__ import division
import os, sys

# this is so we can import from pyhmm
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

from scipy.stats import multivariate_normal as mnorm
import numpy as np
import gmm


def test_constructors():
    T = 10
    S = 3
    L = 3
    D = 3
    cs = np.array([np.ones(L)/L for _ in xrange(S)])
    mus = np.array([[np.ones(D) for _ in xrange(L)] for _ in xrange(S)])
    sigmas = np.array([[np.eye(D) for _ in xrange(L)] for _ in xrange(S)])

    obs = np.ones((T, D))
    lweights = np.ones((T, D))/D

    gmm.update_parameters(cs, mus, sigmas, obs, lweights)

    print "cs = ", cs
    print "mus = ", mus
    print "sigmas = ", sigmas


def test1():
    T = 100
    S = 3
    L = 3
    D = 3

    cs = np.array([np.ones(L)/L for _ in xrange(S)])
    mus = np.array([[0,0,0], [1,0,0], [0,0,1],
                    [0,5,5], [1,5,5], [0,5,6],
                    [5,0,5], [5,1,5], [5,0,6]])
    sigmas = np.array([[np.eye(D) for _ in xrange(L)] for _ in xrange(S)])

    obs = np.array([mnorm.rvs(mean=mus[s][l], cov=sigmas[s][l]) 
        for l in xrange(L) for s in xrange(S) for _ in xrange(T)])

    print 'stop'


if __name__ == '__main__':
    #test_constructors()
    test1()
