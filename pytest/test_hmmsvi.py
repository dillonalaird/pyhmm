from __future__ import division


import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import numpy as np
import hmmsvi as hmm


def test_initialize():
    obs = np.zeros((10,2))
    A_0 = np.ones((2,2))
    emits = np.ones((1,1))
    tau = 1.
    kappa = 0.5
    L = 2
    n = 5
    itr = 5
    hmm.infer(obs, A_0, emits, tau, kappa, L, n, itr)


if __name__ == '__main__':
    test_initialize()
