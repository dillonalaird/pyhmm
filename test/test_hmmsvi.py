from __future__ import division
from test_utils import generate_data

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

from hmmsvi import HMMSVI
from normal_inverse_wishart import NormalInvWishart as NIW

import numpy as np


def test_basic1():
    D = 2
    N = 100
    pi = np.array([0.99, 0.01])
    A = np.array([[0.90, 0.10],
                  [0.10, 0.90]])
    mus = np.array([[0., 0.], [20., 20.]])
    sigmas = np.array([np.eye(2), 2.*np.eye(2)])
    params = [[mus[0], sigmas[0]], [mus[1], sigmas[1]]]

    obs, sts = generate_data(D, N, pi, A, params)

    A_0 = 2*np.ones((2,2))
    emits = [NIW(np.ones(2), np.eye(2), 0.5, 5),
             NIW(20*np.ones(2), 2.*np.eye(2), 0.5, 5)]
    hmm = HMMSVI(obs, A_0, emits, 1., 0.7)
    hmm.infer(10, 10, HMMSVI.metaobs_unif, 20)

    var_x = hmm.full_local_update()


if __name__ == '__main__':
    test_basic1()
