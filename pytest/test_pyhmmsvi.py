from __future__ import division
from test_utils import generate_data
from scipy.spatial.distance import hamming

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

from test_utils import sample_invwishart
from pyhmmsvi import HMMSVI
from normal_inverse_wishart import NormalInvWishart as NIW

import numpy as np


def test_basic1():
    D = 2
    N = 100
    pi = np.array([0.99, 0.01])
    A = np.array([[0.90, 0.10],
                  [0.10, 0.90]])
    mus = np.array([[0., 0.], [20., 20.]])
    sigmas = np.array([np.eye(2), 5.*np.eye(2)])
    kappa = 0.5
    nu = 5

    params = [[mus[0], sigmas[0]], [mus[1], sigmas[1]]]

    print 'label 1 true'
    print mus[0]
    print sample_invwishart(sigmas[0], nu)
    print kappa
    print nu

    print 'label 2 true'
    print mus[1]
    print sample_invwishart(sigmas[1], nu)
    print kappa
    print nu

    obs, sts = generate_data(D, N, pi, A, params)

    A_0 = 2*np.ones((2,2))
    emits = [NIW(np.ones(2), np.eye(2), kappa, nu),
             NIW(20*np.ones(2), 5.*np.eye(2), kappa, nu)]
    hmm = HMMSVI(obs, A_0, emits, 1., 0.7)
    hmm.infer(10, 10, HMMSVI.metaobs_unif, 20)

    var_x = hmm.full_local_update()
    sts_pred = np.argmax(var_x, axis=1)
    print 'hamming distance = ', np.min([hamming(np.array([0,1])[sts_pred], sts),
                                         hamming(np.array([1,0])[sts_pred], sts)])


    print 'label 1 learned'
    print hmm.emits[0].mu_N
    print sample_invwishart(hmm.emits[0].sigma_N, hmm.emits[0].nu_N)
    print hmm.emits[0].kappa_N
    print hmm.emits[0].nu_N

    print 'label 2 learned'
    print hmm.emits[1].mu_N
    print sample_invwishart(hmm.emits[1].sigma_N, hmm.emits[1].nu_N)
    print hmm.emits[1].kappa_N
    print hmm.emits[1].nu_N

if __name__ == '__main__':
    test_basic1()
