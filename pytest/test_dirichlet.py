from __future__ import division
from scipy.special import digamma

import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
    os.pardir))

import numpy as np
import dirichlet as dir


def test_meanfield_sgd_update():
    nat_params_0 = np.ones((2,2)).reshape(2,2)
    nat_params_N = 2.5*np.ones((2,2)).reshape(2,2)
    lrate   = 0.01
    bfactor = 10.
    ess = np.array([[25., 20.],
                    [10., 25.]])

    nat_params_N_true = (1 - lrate)*nat_params_N + lrate*(nat_params_0 +
                                                          bfactor*ess)

    nat_params_N_test = dir.meanfield_sgd_update(nat_params_0, nat_params_N,
                                                 ess, lrate, bfactor)

    np.testing.assert_almost_equal(nat_params_N_test, nat_params_N_true)



def test_expected_sufficient_statistics():
    alphas = np.array([[1.,1.],
                       [1.,1.]])

    ess_true = digamma(alphas) - digamma(np.sum(alphas, axis=1)[:,np.newaxis])
    ess_pred = dir.expected_sufficient_statistics(alphas)
    np.testing.assert_almost_equal(ess_true, ess_pred)

    alphas = np.array([[1.,2.],
                       [3.,4.]])

    ess_true = digamma(alphas) - digamma(np.sum(alphas, axis=1)[:,np.newaxis])
    ess_test = dir.expected_sufficient_statistics(alphas)
    np.testing.assert_almost_equal(ess_test, ess_true)


def test_sufficient_statistics1():
    N = 100
    dists = np.array([[1.,0.], [0.,1.]])
    expected_states = np.array([dists[round(i/N)] for i in xrange(N)])

    ss_true = np.zeros((2,2))
    for i in xrange(1,expected_states.shape[0]):
        ss_true += np.outer(expected_states[i-1],expected_states[i])

    ss = dir.sufficient_statistics(expected_states)

    np.testing.assert_almost_equal(ss, ss_true, decimal=1)


def test_sufficient_statistics2():
    N = 100
    dists = np.array([[0.5,0.5], [0.5,0.5]])
    expected_states = np.array([dists[round(i/N)] for i in xrange(N)])

    ss_true = np.zeros((2,2))
    for i in xrange(1,expected_states.shape[0]):
        ss_true += np.outer(expected_states[i-1],expected_states[i])

    ss = dir.sufficient_statistics(expected_states)

    np.testing.assert_almost_equal(ss, ss_true, decimal=1)


def test_sufficient_statistics3():
    N = 100
    dists = np.array([[0.75,0.25], [0.99,0.01]])
    expected_states = np.array([dists[round(i/N)] for i in xrange(N)])

    ss_true = np.zeros((2,2))
    for i in xrange(1,expected_states.shape[0]):
        ss_true += np.outer(expected_states[i-1],expected_states[i])

    ss_test = dir.sufficient_statistics(expected_states)

    np.testing.assert_almost_equal(ss_test, ss_true, decimal=1)


if __name__ == '__main__':
    test_meanfield_sgd_update()
    #test_expected_sufficient_statistics()
    #test_sufficient_statistics1()
    #test_sufficient_statistics2()
    #test_sufficient_statistics3()
