from __future__ import division
import os, sys, inspect

# this is so we can import from pyhmm
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import numpy as np
import gmm


def test_constructors():
    T = 10
    S = 5
    L = 3
    D = 3
    cs = np.array([np.ones(L)/L for _ in xrange(S)])
    mus = np.array([[np.ones(D) for _ in xrange(L)] for _ in xrange(S)])
    sigmas = np.array([[np.eye(D) for _ in xrange(L)] for _ in xrange(S)])

    obs = np.ones((T, D))
    lweights = np.ones((T, D))/D

    gmm.update_parameters(cs, mus, sigmas, obs, lweights)


if __name__ == '__main__':
    test_constructors()
