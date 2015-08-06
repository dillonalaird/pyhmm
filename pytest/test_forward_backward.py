from __future__ import division
import os, sys

# this is so we can import from pyhmm
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

from scipy.stats import norm
import numpy as np
import forward_backward as fb


def forward_messages_corr(pi, A, lliks):
    Al = np.log(A)
    lalpha = np.zeros_like(lliks)
    lalpha[0,:] = np.log(pi) + lliks[0]
    for t in xrange(lalpha.shape[0]-1):
        lalpha[t+1] = np.logaddexp.reduce(lalpha[t] + Al.T, axis=1) + lliks[t+1]

    return lalpha


def backward_messages_corr(A, lliks):
    Al = np.log(A)
    lbeta = np.zeros_like(lliks)
    for t in xrange(lbeta.shape[0]-2,-1,-1):
        np.logaddexp.reduce(Al + lbeta[t+1] + lliks[t+1], axis=1, out=lbeta[t])

    return lbeta


def test_simple_fb():
    N = 100
    pi = np.array([0.99, 0.01])
    # we have 1 transition from 0 to 1
    A = np.array([[(N-1)/N, 1-(N-1)/N],
                  [1e-7, 1.0-1e-7]])

    obs = np.empty(N)
    dist = np.array([[0, 1], [5, 1]])
    obs = np.array([norm.rvs(dist[np.round(i/N),0], dist[np.round(i/N),1])
                    for i in xrange(1, N+1)])

    # lliks needs to have 2 dimensions
    lliks = np.array([[norm.logpdf(ob, dist[np.round((i+1)/N),0],
        dist[np.round((i+1)/N),1])] for i,ob in enumerate(obs)])

    lliks = np.array([[norm.logpdf(ob, dist[0,0], dist[0,1]),
                       norm.logpdf(ob, dist[1,0], dist[1,1])]
                       for i,ob in enumerate(obs)])


    lalpha = fb.forward_msgs(pi, A, lliks)
    lalpha_corr = forward_messages_corr(pi, A, lliks)

    lbeta = fb.backward_msgs(A, lliks)
    lbeta_corr = backward_messages_corr(A, lliks)

    np.testing.assert_almost_equal(lalpha, lalpha_corr)
    np.testing.assert_almost_equal(lbeta, lbeta_corr)


def test_expected_states_and_transcount():
    N = 100
    pi = np.array([0.99, 0.01])
    # we have 1 transition from 0 to 1
    A = np.array([[(N-1)/N, 1-(N-1)/N],
                  [1-(N-1)/N, (N-1)/N]])

    obs = np.empty(N)
    dist = np.array([[0, 1], [5, 1]])
    obs = np.array([norm.rvs(dist[np.round(i/N),0], dist[np.round(i/N),1])
        for i in xrange(1, N+1)])

    # lliks needs to have 2 dimensions
    lliks = np.array([[norm.logpdf(ob, dist[np.round((i+1)/N),0],
        dist[np.round((i+1)/N),1])] for i,ob in enumerate(obs)])

    lliks = np.array([[norm.logpdf(ob, dist[0,0], dist[0,1]),
                       norm.logpdf(ob, dist[1,0], dist[1,1])]
                       for i,ob in enumerate(obs)])


    lalpha = fb.forward_msgs(pi, A, lliks)
    lbeta = fb.backward_msgs(A, lliks)

    lexpected_states, expected_transcounts = fb.expected_statistics(pi, A, lliks,
            lalpha, lbeta)
    expected_states = np.exp(lexpected_states)
    expected_states = expected_states / \
            np.sum(expected_states, axis=1)[:,np.newaxis]
    expected_transcounts /= np.sum(expected_transcounts, axis=1)[:,np.newaxis]

    expected_transcounts_corr = np.array([[1., 0.],
                                          [0., 1.]])
    probs = np.array([[1., 0.], [0., 1.]])
    expected_states_corr = np.array([probs[np.round(i/N)] for i in xrange(1, N+1)])

    np.testing.assert_almost_equal(expected_states, expected_states_corr,
            decimal=1)
    np.testing.assert_almost_equal(expected_transcounts,
            expected_transcounts_corr, decimal=1)


if __name__ == '__main__':
    #test_simple_fb()
    test_expected_states_and_transcount()
