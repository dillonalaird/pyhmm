from __future__ import division


import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import numpy as np
import hmmsvi as hmm


def test_initialize():
    D = 2
    S = 2
    obs = np.zeros((10,2))
    A_0 = np.ones((2,2))

    # just send these in in a list
    emit1 = np.zeros((2+3,2))
    emit1[:2,:2] = np.eye(2)
    emit1[2,:]   = np.ones(2)
    emit1[3,0]   = 0.5
    emit1[4,0]   = 5

    emit2 = np.zeros((2+3,2))
    emit2[:2,:2] = 5.*np.eye(2)
    emit2[2,:]   = 10.*np.ones(2)
    emit2[3,0]   = 0.5
    emit2[4,0]   = 5

    emits = np.vstack((emit1, emit2))

    #emits_flat = np.zeros(S*(D*D + 3*D))
    #offset1 = D*D + 3*D
    #offset2 = D + 3
    #for s in range(S):
    #    emits_flat[(s*offset1):(s*offset1 + D*D)] = emits[(s*offset2):(s*offset2 + D)].flatten()
    #    emits_flat[(s*offset1 + D*D):(s*offset1 + D*D + D)] = emits[(s*offset2 + D),:].flatten()
    #    emits_flat[(s*offset1 + D*D + D)] = emits[(s*offset2 + D + 1),0].flatten()
    #    emits_flat[(s*offset1 + D*D + 2*D)] = emits[(s*offset2 + D + 2),0].flatten()

    #    print emits[(s*offset2):(s*offset2 + D)]
    #    print emits[(s*offset2 + D),:]
    #    print emits[(s*offset2 + D + 1),0]
    #    print emits[(s*offset2 + D + 2),0]

    tau = 1.
    kappa = 0.5
    L = 2
    n = 5
    itr = 5
    hmm.infer(obs, A_0, emits, tau, kappa, L, n, itr)


if __name__ == '__main__':
    test_initialize()
