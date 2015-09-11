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
    print 'BEFORE'
    print A_0
    print emits

    tau = 1.
    kappa = 0.5
    L = 2
    n = 5
    itr = 5
    A_N, emits_N = hmm.infer(obs, A_0, emits, tau, kappa, L, n, itr)
    print 'AFTER'
    print A_N
    print emits_N


def test_basic1():
    D = 2
    S = 2
    obs = []
    obs_i = np.array([[0., 0.], [5., 5.]])
    for i in xrange(20):
        obs.append(obs_i[np.round(i/20)])
    obs = np.array(obs)

    A_0 = np.ones((2,2))

    emit1 = np.zeros((2+3,2))
    emit1[:2,:2] = np.eye(2)
    emit1[2,:]   = np.ones(2)
    emit1[3,0]   = 0.5
    emit1[4,0]   = 5

    emit2 = np.zeros((2+3,2))
    emit2[:2,:2] = np.eye(2)
    emit2[2,:]   = 5.*np.ones(2)
    emit2[3,0]   = 0.5
    emit2[4,0]   = 5

    emits = np.vstack((emit1, emit2))
    print 'BEFORE'
    print A_0
    print emits

    tau = 1.
    kappa = 0.5
    L = 2
    n = 5
    itr = 5
    A_N, emits_N = hmm.infer(obs, A_0, emits, tau, kappa, L, n, itr)

    print 'AFTER'
    print A_N
    print emits_N


def test_basic2():
    D = 2
    S = 2
    obs = []
    obs_i = np.array([[0., 0.], [5., 5.]])
    for i in xrange(20):
        obs.append(obs_i[np.round(i/20)])
    obs = np.array(obs)

    A_0 = np.ones((2,2))

    emit1 = np.zeros((2+3,2))
    emit1[:2,:2] = np.eye(2)
    emit1[2,:]   = 2.*np.ones(2)
    emit1[3,0]   = 0.5
    emit1[4,0]   = 5

    emit2 = np.zeros((2+3,2))
    emit2[:2,:2] = np.eye(2)
    emit2[2,:]   = 6.*np.ones(2)
    emit2[3,0]   = 0.5
    emit2[4,0]   = 5

    emits = np.vstack((emit1, emit2))
    print 'BEFORE'
    print A_0
    print emits

    print 'TRUE'
    emit1 = np.zeros((2+3,2))
    emit1[:2,:2] = np.eye(2)
    emit1[2,:]   = np.zeros(2)
    emit1[3,0]   = 0.5
    emit1[4,0]   = 5

    emit2 = np.zeros((2+3,2))
    emit2[:2,:2] = np.eye(2)
    emit2[2,:]   = 5.*np.ones(2)
    emit2[3,0]   = 0.5
    emit2[4,0]   = 5

    emits = np.vstack((emit1, emit2))

    tau = 1.
    kappa = 0.5
    L = 2
    n = 5
    itr = 5
    A_N, emits_N = hmm.infer(obs, A_0, emits, tau, kappa, L, n, itr)

    print 'AFTER'
    print A_N
    print emits_N


def test_basic3():
    D = 2
    S = 2
    N = 1000
    A_true = np.array([[0.3, 0.7],
                       [0.1, 0.9]])
    obs = []
    st = 0
    sts = [st]
    params = [{'mu': np.ones(2.), 'sigma': np.array([[5., 2.], [3., 5.]])},
              {'mu': np.array([15., 20.]), 'sigma': np.array([[2., 0.5], [1., 3.]])}]
    for i in xrange(N):
        obs.append(np.random.multivariate_normal(params[st]['mu'],
                                                 params[st]['sigma']))
        st = np.random.choice(2, 1, p=A_true[st,:])[0]
        sts.append(st)
    obs = np.array(obs, order='C')

    A_0 = np.ones((2,2))

    emit1 = np.zeros((2+3,2))
    emit1[:2,:2] = np.eye(2)
    emit1[2,:]   = np.zeros(2)
    emit1[3,0]   = 0.5
    emit1[4,0]   = 5

    emit2 = np.zeros((2+3,2))
    emit2[:2,:2] = np.eye(2)
    # off by one
    emit2[2,:]   = 5.*np.ones(2)
    emit2[3,0]   = 0.5
    emit2[4,0]   = 5

    emits = np.vstack((emit1, emit2))
    print 'True'
    print A_true
    print params[0]['mu']
    print params[0]['sigma']
    print params[1]['mu']
    print params[1]['sigma']

    tau = 1.
    kappa = 0.5
    L = 2
    n = 5
    itr = 5
    A_N, emits_N = hmm.infer(obs, A_0, emits, tau, kappa, L, n, itr)

    print 'AFTER'
    print A_N
    print emits_N[2,:]
    print emits_N[:2,:2]
    print emits_N[7,:]
    print emits_N[5:7,:]

if __name__ == '__main__':
    #test_initialize()
    #test_basic1()
    #test_basic2()
    test_basic3()
