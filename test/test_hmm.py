from __future__ import division
from scipy.stats import multivariate_normal as mnorm
from matplotlib import pyplot as plt

import numpy as np
import scipy.stats as stats
import scipy
import os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import forward_backward as fb
import normal_invwishart as niw
import dirichlet as dir


def _sample_niw(mu, lmbda, kappa, nu):
    lmbda = _sample_invwishart(lmbda, nu)
    mu = np.random.multivariate_normal(mu, lmbda/kappa)
    return mu, lmbda


def _sample_invwishart(S, nu):
    n = S.shape[0]
    chol = np.linalg.cholesky(S)

    if (nu <= 81+n) and (nu == np.round(nu)):
        x = np.random.randn(nu, n)
    else:
        x = np.diag(np.sqrt(np.atleast_1d(stats.chi2.rvs(nu-np.arange(n)))))
        x[np.triu_indices_from(x,1)] = np.random.randn(n*(n-1)//2)
    R = np.linalg.qr(x, 'r')
    T = scipy.linalg.solve_triangular(R.T, chol.T, lower=True).T
    return np.dot(T, T.T)


def standard_to_natural(mu_0, sigma_0, kappa_0, nu_0):
    n1 = kappa_0*mu_0
    n2 = kappa_0
    n3 = sigma_0 + kappa_0*np.outer(mu_0, mu_0)
    n4 = nu_0 + 2 + mu_0.shape[0]
    return n1, n2, n3, n4


def natural_to_standard(n1, n2, n3, n4):
    kappa_0 = n2
    mu_0 = n1/n2
    sigma_0 = n3 - kappa_0*np.outer(mu_0, mu_0)
    nu_0 = n4 - 2 - mu_0.shape[0]
    return mu_0, sigma_0, kappa_0, nu_0


def generate_data(D, N, pi, A, params):
    s = np.random.choice(D, 1, p=pi)[0]
    obs = [mnorm.rvs(params[s][0], params[s][1])]
    for i in xrange(N-1):
        s = np.random.choice(2, 1, p=A[s,:])[0]
        obs.append(mnorm.rvs(params[s][0], params[s][1]))

    return np.array(obs)


def test_basic1():
    N = 100
    D = 2
    pi = np.array([0.999, 0.001])
    A = np.array([[0.9, 0.1],
                  [0.1, 0.9]])
    mus_0 = np.array([[0.,0.], [20.,20.]])
    sigmas_0 = np.array([np.eye(2), 2*np.eye(2)])
    kappa_0 = 0.5
    nu_0 = 5

    mu_1, sigma_1 = _sample_niw(mus_0[0], sigmas_0[0], kappa_0, nu_0)
    mu_2, sigma_2 = _sample_niw(mus_0[1], sigmas_0[1], kappa_0, nu_0)

    print 'label 1 before'
    print mu_1
    print sigma_1

    print 'label 2 before'
    print mu_2
    print sigma_2

    params = [[mu_1, sigma_1], [mu_2, sigma_2]]

    obs = generate_data(D, N, pi, A, params)

    n1s = [kappa_0*mus_0[0],
           kappa_0,
           sigmas_0[0] + kappa_0*np.outer(mus_0[0],mus_0[0]),
           nu_0 + D + 2]
    n2s = [kappa_0*mus_0[1],
           kappa_0,
           sigmas_0[1] + kappa_0*np.outer(mus_0[1],mus_0[1]),
           nu_0 + D + 2]

    lliks1 = niw.expected_log_likelihood(obs, mus_0[0], sigmas_0[0], kappa_0, nu_0)
    lliks2 = niw.expected_log_likelihood(obs, mus_0[1], sigmas_0[1], kappa_0, nu_0)
    lliks = np.vstack((lliks1, lliks2)).T.copy(order='C')

    lalpha = fb.forward_msgs(pi, A, lliks)
    lbeta = fb.backward_msgs(A, lliks)

    lexpected_states = lalpha + lbeta
    lexpected_states -= np.max(lexpected_states, axis=1)[:,np.newaxis]
    expected_states  = np.exp(lexpected_states)
    expected_states /= np.sum(expected_states, axis=1)[:,np.newaxis]

    s1s = np.sum(expected_states, axis=0)
    s2s = np.array([np.sum(obs*expected_states[:,i,np.newaxis], axis=0)
                    for i in xrange(2)])
    s3s = np.array([np.sum([np.outer(obs[i],obs[i])*expected_states[i,0]
                            for i in xrange(obs.shape[0])], axis=0),
                    np.sum([np.outer(obs[i],obs[i])*expected_states[i,1]
                            for i in xrange(obs.shape[0])], axis=0)])

    n11, n12, n13, n14 = niw.meanfield_update(n1s[0], n1s[1], n1s[2], n1s[3],
                                              s1s[0], s2s[0], s3s[0])
    n21, n22, n23, n24 = niw.meanfield_update(n2s[0], n2s[1], n2s[2], n2s[3],
                                              s1s[1], s2s[1], s3s[1])

    # Note might need to add small positive to diagonal
    mu_0, sigma_0, kappa_0, nu_0 = natural_to_standard(n11, n12, n13, n14)
    mu_1, sigma_1 = _sample_niw(mu_0, sigma_0, kappa_0, nu_0)
    print 'label 1 after'
    print mu_1
    print sigma_1

    mu_0, sigma_0, kappa_0, nu_0 = natural_to_standard(n21, n22, n23, n24)
    mu_2, sigma_2 = _sample_niw(mu_0, sigma_0, kappa_0, nu_0)
    print 'label 2 after'
    print mu_2
    print sigma_2


def test_basic2():
    N = 100
    D = 2
    pi = np.array([0.999, 0.001])
    A = np.array([[0.9, 0.1],
                  [0.1, 0.9]])
    mus_0 = np.array([[0.,0.], [20.,20.]])
    sigmas_0 = np.array([np.eye(2), 5*np.eye(2)])
    kappa_0 = 0.5
    nu_0 = 5

    mu_1, sigma_1 = _sample_niw(mus_0[0], sigmas_0[0], kappa_0, nu_0)
    mu_2, sigma_2 = _sample_niw(mus_0[1], sigmas_0[1], kappa_0, nu_0)

    params = [[mu_1, sigma_1], [mu_2, sigma_2]]

    obs = generate_data(D, N, pi, A, params)

    plt.scatter(obs[:,0], obs[:,1])
    plt.show()

    mus_N = np.array([np.mean(obs, axis=0), np.mean(obs, axis=0)])
    sigmas_N = 0.75*np.array([np.cov(obs.T), np.cov(obs.T)])

    n1s_0 = [kappa_0*mus_N[0],
             kappa_0,
             sigmas_N[0] + kappa_0*np.outer(mus_N[0],mus_N[0]),
             nu_0 + D + 2]
    n2s_0 = [kappa_0*mus_N[1],
             kappa_0,
             sigmas_N[1] + kappa_0*np.outer(mus_N[1],mus_N[1]),
             nu_0 + D + 2]

    n1s_N = n1s_0[:]
    n2s_N = n2s_0[:]

    iters = 20

    print 'label 1 true'
    print mus_0[0]
    print sigmas_0[0]

    print 'label 2 true'
    print mus_0[1]
    print sigmas_0[1]

    for _ in xrange(iters):

        mu_1N, sigma_1N, kappa_1N, nu_1N = natural_to_standard(n1s_N[0], n1s_N[1],
                                                               n1s_N[2], n1s_N[3])
        mu_2N, sigma_2N, kappa_2N, nu_2N = natural_to_standard(n2s_N[0], n2s_N[1],
                                                               n2s_N[2], n2s_N[3])

        lliks1 = niw.expected_log_likelihood(obs, mu_1N, sigma_1N, kappa_1N, nu_1N)
        lliks2 = niw.expected_log_likelihood(obs, mu_2N, sigma_2N, kappa_2N, nu_2N)
        lliks = np.vstack((lliks1, lliks2)).T.copy(order='C')

        lalpha = fb.forward_msgs(pi, A, lliks)
        lbeta = fb.backward_msgs(A, lliks)

        lexpected_states = lalpha + lbeta
        lexpected_states -= np.max(lexpected_states, axis=1)[:,np.newaxis]
        expected_states  = np.exp(lexpected_states)
        expected_states /= np.sum(expected_states, axis=1)[:,np.newaxis]

        s11, s12, s13 = niw.expected_sufficient_statistics(obs,
                expected_states[:,0].copy(order='C'))
        s21, s22, s23 = niw.expected_sufficient_statistics(obs,
                expected_states[:,1].copy(order='C'))
        s1s = np.array([s11, s21])
        s2s = np.array([s12, s22])
        s3s = np.array([s13, s23])

        n11, n12, n13, n14 = niw.meanfield_update(n1s_0[0], n1s_0[1], n1s_0[2],
                                                  n1s_0[3], s1s[0], s2s[0],
                                                  s3s[0])
        n21, n22, n23, n24 = niw.meanfield_update(n2s_0[0], n2s_0[1], n2s_0[2],
                                                  n2s_0[3], s1s[1], s2s[1],
                                                  s3s[1])
        n1s_N = [n11, n12, n13, n14]
        n2s_N = [n21, n22, n23, n24]

    mu_0, sigma_0, kappa_0, nu_0 = natural_to_standard(n11, n12, n13, n14)
    mu_1, sigma_1 = _sample_niw(mu_0, sigma_0, kappa_0, nu_0)
    print 'label 1 learned'
    print mu_1
    print sigma_1

    mu_0, sigma_0, kappa_0, nu_0 = natural_to_standard(n21, n22, n23, n24)
    mu_2, sigma_2 = _sample_niw(mu_0, sigma_0, kappa_0, nu_0)
    print 'label 2 learned'
    print mu_2
    print sigma_2


def test_basic3():
    N = 100
    D = 2
    pi = np.array([0.999, 0.001])
    A = np.array([[0.9, 0.1],
                  [0.1, 0.9]])

    mus_0 = np.array([[0.,0.], [20.,20.]])
    sigmas_0 = np.array([np.eye(2), 5*np.eye(2)])
    kappa_0 = 0.5
    nu_0 = 5

    mu_1, sigma_1 = _sample_niw(mus_0[0], sigmas_0[0], kappa_0, nu_0)
    mu_2, sigma_2 = _sample_niw(mus_0[1], sigmas_0[1], kappa_0, nu_0)

    params = [[mu_1, sigma_1], [mu_2, sigma_2]]

    obs = generate_data(D, N, pi, A, params)

    #plt.scatter(obs[:,0], obs[:,1])
    #plt.show()

    A_0 = 2*np.ones((A.shape[0], A.shape[0]))
    mus_N = np.array([np.mean(obs, axis=0), np.mean(obs, axis=0)])
    sigmas_N = 0.75*np.array([np.cov(obs.T), np.cov(obs.T)])

    A_nat_0 = A_0 - 1
    n1s_0 = [kappa_0*mus_N[0],
             kappa_0,
             sigmas_N[0] + kappa_0*np.outer(mus_N[0],mus_N[0]),
             nu_0 + D + 2]
    n2s_0 = [kappa_0*mus_N[1],
             kappa_0,
             sigmas_N[1] + kappa_0*np.outer(mus_N[1],mus_N[1]),
             nu_0 + D + 2]

    A_nat_N = A_nat_0[:]
    n1s_N = n1s_0[:]
    n2s_N = n2s_0[:]

    iters = 20

    print 'A true'
    print A

    print 'label 1 true'
    #print mus_0[0]
    #print sigmas_0[0]
    print mu_1
    print sigma_1

    print 'label 2 true'
    #print mus_0[1]
    #print sigmas_0[1]
    print mu_2
    print sigma_2

    for _ in xrange(iters):
        mu_1N, sigma_1N, kappa_1N, nu_1N = natural_to_standard(n1s_N[0], n1s_N[1],
                                                               n1s_N[2], n1s_N[3])
        mu_2N, sigma_2N, kappa_2N, nu_2N = natural_to_standard(n2s_N[0], n2s_N[1],
                                                               n2s_N[2], n2s_N[3])

        lliks1 = niw.expected_log_likelihood(obs, mu_1N, sigma_1N, kappa_1N, nu_1N)
        lliks2 = niw.expected_log_likelihood(obs, mu_2N, sigma_2N, kappa_2N, nu_2N)
        lliks = np.vstack((lliks1, lliks2)).T.copy(order='C')

        lA_mod = dir.expected_sufficient_statistics(A_nat_N + 1)
        A_mod = np.exp(lA_mod)

        lalpha = fb.forward_msgs(pi, A, lliks)
        lbeta = fb.backward_msgs(A, lliks)

        lexpected_states, expected_transcounts = fb.expected_statistics(pi,
                A_mod, lliks, lalpha, lbeta)

        #lexpected_states = lalpha + lbeta
        #lexpected_states -= np.max(lexpected_states, axis=1)[:,np.newaxis]
        expected_states  = np.exp(lexpected_states)
        expected_states /= np.sum(expected_states, axis=1)[:,np.newaxis]
        expected_transcounts /= np.sum(expected_transcounts,
                                       axis=1)[:,np.newaxis]


        A_ss = np.log(expected_transcounts)
        A_nat_N = dir.meanfield_update(A_nat_0, A_ss)

        s11, s12, s13 = niw.expected_sufficient_statistics(obs,
                expected_states[:,0].copy(order='C'))
        s21, s22, s23 = niw.expected_sufficient_statistics(obs,
                expected_states[:,1].copy(order='C'))
        s1s = np.array([s11, s21])
        s2s = np.array([s12, s22])
        s3s = np.array([s13, s23])

        n11, n12, n13, n14 = niw.meanfield_update(n1s_0[0], n1s_0[1], n1s_0[2],
                                                  n1s_0[3], s1s[0], s2s[0],
                                                  s3s[0])
        n21, n22, n23, n24 = niw.meanfield_update(n2s_0[0], n2s_0[1], n2s_0[2],
                                                  n2s_0[3], s1s[1], s2s[1],
                                                  s3s[1])

        n1s_N = [n11, n12, n13, n14]
        n2s_N = [n21, n22, n23, n24]

    print 'A learned'
    A_sample = np.array([stats.dirichlet.rvs(A_nat_N[i,:] + 1, size=1)[0]
                         for i in xrange(A.shape[0])])
    print A_sample

    mu_0, sigma_0, kappa_0, nu_0 = natural_to_standard(n11, n12, n13, n14)
    mu_1, sigma_1 = _sample_niw(mu_0, sigma_0, kappa_0, nu_0)
    print 'label 1 learned'
    print mu_1
    print sigma_1

    mu_0, sigma_0, kappa_0, nu_0 = natural_to_standard(n21, n22, n23, n24)
    mu_2, sigma_2 = _sample_niw(mu_0, sigma_0, kappa_0, nu_0)
    print 'label 2 learned'
    print mu_2
    print sigma_2


if __name__ == '__main__':
    #test_basic1()
    #test_basic2()
    test_basic3()