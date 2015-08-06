from __future__ import division
from scipy.stats import multivariate_normal as mnorm
from matplotlib import pyplot as plt
from test_utils import sample_niw, responsibilities, natural_to_standard, \
                       standard_to_natural

import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import numpy as np
import normal_invwishart as niw


def test_constructors():
    D = 3

    n1 = np.array([1,2,3]).astype(np.double)
    n2 = 4.
    n3 = np.array([[5,6,7],[8,9,10],[11,12,13]]).astype(np.double)
    n4 = 14.
    s1 = 1.
    s2 = np.array([1,1,1]).astype(np.double)
    s3 = np.array([[1,1,1],[1,1,1],[1,1,1]]).astype(np.double)

    niw.meanfield_update(n1, n2, n3, n4, s1, s2, s3)

    print 'after n1 = ', n1
    print 'after n2 = ', n2
    print 'after n3 = ', n3
    print 'after n4 = ', n4


def test_update1():
    D = 3

    n1 = np.array([1,2,3]).astype(np.double)
    n2 = 4.
    n3 = np.array([[5,6,7],[8,9,10],[11,12,13]]).astype(np.double)
    n4 = 14.
    s1 = 1.
    s2 = np.array([1,1,1]).astype(np.double)
    s3 = np.array([[1,1,1],[1,1,1],[1,1,1]]).astype(np.double)

    n1, n2, n3, n4 = niw.meanfield_update(n1, n2, n3, n4, s1, s2, s3)

    print 'after n1 = ', n1
    print 'after n2 = ', n2
    print 'after n3 = ', n3
    print 'after n4 = ', n4


def test_update2():
    N = 100
    mus_0 = np.array([[0,0], [20,20]])
    sigmas_0 = np.array([np.eye(2), 5*np.eye(2)])
    kappa_0 = 0.5
    nu_0 = 5

    mu_1, sigma_1 = sample_niw(mus_0[0], sigmas_0[0], kappa_0, nu_0)
    mu_2, sigma_2 = sample_niw(mus_0[1], sigmas_0[1], kappa_0, nu_0)
    params = [[mu_1, sigma_1], [mu_2, sigma_2]]

    print 'sampled params'
    print '\tmu_1 = ', mu_1
    print '\tsigma_1 = ', (1/kappa_0)*sigma_1
    print '\tmu_2 = ', mu_2
    print '\tsigma_2 = ', (1/kappa_0)*sigma_2

    obs = np.array([mnorm.rvs(params[int(np.round(i/N))][0],
                              params[int(np.round(i/N))][1])
                    for i in xrange(1,N+1)])

    #plt.scatter(obs[:,0], obs[:,1])
    #plt.show()

    rs1 = responsibilities(mus_0[0], sigmas_0[0], kappa_0, nu_0, obs)
    rs2 = responsibilities(mus_0[1], sigmas_0[1], kappa_0, nu_0, obs)
    rs = np.vstack((rs1, rs2)).T
    rs = np.exp(rs)
    rs /= np.sum(rs, axis=1)[:,np.newaxis]

    s1s = np.sum(rs, axis=0)
    s2s = np.array([np.sum(obs*rs[:,i,np.newaxis], axis=0) for i in xrange(2)])
    s3s = np.array([np.sum([np.outer(obs[i],obs[i])*rs[i,0]
                            for i in xrange(obs.shape[0])],axis=0),
                    np.sum([np.outer(obs[i],obs[i])*rs[i,1]
                            for i in xrange(obs.shape[0])],axis=0)])

    n1s = np.array([kappa_0*mus_0[0], kappa_0*mus_0[1]])
    n2s = np.array([kappa_0, kappa_0])
    n3s = np.array([sigmas_0[0] + kappa_0*np.outer(mus_0[0], mus_0[0]),
                    sigmas_0[1] + kappa_0*np.outer(mus_0[1], mus_0[1])])
    n4s = np.array([nu_0 + 2 + 2, nu_0 + 2 + 2])

    #print 'label 1 before'
    #print '\tmu_0 = ', mus_0[0]
    #print '\tsigma_0 = ', sigmas_0[0]
    #print '\tkappa_0 = ', kappa_0
    #print '\tnu_0 = ', nu_0

    n1, n2, n3, n4 = niw.meanfield_update(n1s[0], n2s[0], n3s[0], n4s[0],
                                          s1s[0], s2s[0], s3s[0])

    mu_0, sigma_0, kappa_0, nu_0 = natural_to_standard(n1, n2, n3, n4)

    mu_1, sigma_1 = sample_niw(mu_0, sigma_0, kappa_0, nu_0)

    #print 'label 1 after'
    #print '\tmu_0 = ', mu_0
    #print '\tsigma_0 = ', sigma_0
    #print '\tkappa_0 = ', kappa_0
    #print '\tnu_0 = ', nu_0

    #print 'label 2 before'
    #print '\tmu_0 = ', mus_0[1]
    #print '\tsigma_0 = ', sigmas_0[1]
    #print '\tkappa_0 = ', kappa_0
    #print '\tnu_0 = ', nu_0

    n1, n2, n3, n4 = niw.meanfield_update(n1s[1], n2s[1], n3s[1], n4s[1],
                                          s1s[1], s2s[1], s3s[1])

    mu_0, sigma_0, kappa_0, nu_0 = natural_to_standard(n1, n2, n3, n4)

    #print 'label 2 after'
    #print '\tmu_0 = ', mu_0
    #print '\tsigma_0 = ', sigma_0
    #print '\tkappa_0 = ', kappa_0
    #print '\tnu_0 = ', nu_0

    mu_2, sigma_2 = sample_niw(mu_0, sigma_0, kappa_0, nu_0)

    print 'resampled params'
    print '\tmu_1 = ', mu_1
    print '\tsigma_1 = ', sigma_1
    print '\tmu_2 = ', mu_2
    print '\tsigma_2 = ', sigma_2


def test_meanfield_sgd_update():
    D = 2
    mu_0 = 1.0*np.ones(2)
    sigma_0 = 1.0*np.eye(2)
    kappa_0 = 0.5
    nu_0 = 5.

    mu_N = 1.5*np.ones(2)
    sigma_N = 1.5*np.eye(2)
    kappa_N = 10.5
    nu_N = 10.

    lrate = 0.1
    bfactor = 10.

    n1_0, n2_0, n3_0, n4_0 = standard_to_natural(mu_0, sigma_0, kappa_0, nu_0)
    n1_N, n2_N, n3_N, n4_N = standard_to_natural(mu_N, sigma_N, kappa_N, nu_N)

    nat_params_0 = np.zeros((D+3,D))
    nat_params_0[:D,:D] = n3_0.copy()
    nat_params_0[D,:]   = n1_0.copy()
    nat_params_0[D+1,0] = n2_0
    nat_params_0[D+2,0] = n4_0

    nat_params_N = np.zeros((D+3,D))
    nat_params_N[:D,:D] = n3_N.copy()
    nat_params_N[D,:]   = n1_N.copy()
    nat_params_N[D+1,0] = n2_N
    nat_params_N[D+2,0] = n4_N

    s1 = 10.
    s2 = 5.0*np.ones(2)
    s3 = 5.0*np.eye(2)

    n1_N_true = (1 - lrate)*n1_N + lrate*(n1_0 + bfactor*s2)
    n2_N_true = (1 - lrate)*n2_N + lrate*(n2_0 + bfactor*s1)
    n3_N_true = (1 - lrate)*n3_N + lrate*(n3_0 + bfactor*s3)
    n4_N_true = (1 - lrate)*n4_N + lrate*(n4_0 + bfactor*s1)

    nat_params_N_test = niw.meanfield_sgd_update(nat_params_0, nat_params_N, s1,
                                                 s2, s3, lrate, bfactor)

    np.testing.assert_almost_equal(nat_params_N_test[D,:],   n1_N_true)
    np.testing.assert_almost_equal(nat_params_N_test[D+1,0], n2_N_true)
    np.testing.assert_almost_equal(nat_params_N_test[:D,:D], n3_N_true)
    np.testing.assert_almost_equal(nat_params_N_test[D+2,0], n4_N_true)


def test_expected_log_likelihood():
    N = 10
    mus_0 = np.array([[0.,0.], [20.,20.]])
    sigmas_0 = np.array([np.eye(2), np.eye(2)])
    kappa_0 = 0.5
    nu_0 = 5

    mu_1, sigma_1 = sample_niw(mus_0[0], sigmas_0[0], kappa_0, nu_0)
    mu_2, sigma_2 = sample_niw(mus_0[1], sigmas_0[1], kappa_0, nu_0)
    params = [[mu_1, sigma_1], [mu_2, sigma_2]]

    #print 'sampled params'
    #print '\tmu_1 = ', mu_1
    #print '\tsigma_1 = ', (1/kappa_0)*sigma_1
    #print '\tmu_2 = ', mu_2
    #print '\tsigma_2 = ', (1/kappa_0)*sigma_2

    obs = np.array([mnorm.rvs(params[int(np.round(i/N))][0],
                              (1/kappa_0)*params[int(np.round(i/N))][1])
                    for i in xrange(1,N+1)]).astype(np.float64)

    #plt.scatter(obs[:,0], obs[:,1])
    #plt.show()

    rs1 = niw.expected_log_likelihood(obs, mus_0[0], sigmas_0[0], kappa_0, nu_0)
    rs2 = niw.expected_log_likelihood(obs, mus_0[1], sigmas_0[1], kappa_0, nu_0)

    rs1_true = responsibilities(mus_0[0], sigmas_0[0], kappa_0, nu_0, obs)
    rs2_true = responsibilities(mus_0[1], sigmas_0[1], kappa_0, nu_0, obs)

    np.testing.assert_almost_equal(rs1, rs1_true)
    np.testing.assert_almost_equal(rs2, rs2_true)


def test_log_likelihood():
    N = 10
    mus_0 = np.array([[0.,0.], [20.,20.]])
    sigmas_0 = np.array([np.eye(2), np.eye(2)])
    kappa_0 = 0.5
    nu_0 = 5

    mu_1, sigma_1 = sample_niw(mus_0[0], sigmas_0[0], kappa_0, nu_0)
    mu_2, sigma_2 = sample_niw(mus_0[1], sigmas_0[1], kappa_0, nu_0)
    params = [[mu_1, sigma_1], [mu_2, sigma_2]]

    obs = np.array([mnorm.rvs(params[int(np.round(i/N))][0],
                              params[int(np.round(i/N))][1])
                    for i in xrange(1,N+1)]).astype(np.float64)

    lliks_true = np.array([[mnorm.logpdf(ob, mu_1, sigma_1),
                            mnorm.logpdf(ob, mu_2, sigma_2)] for ob in obs])
    lliks_test1 = niw.log_likelihood(obs, mu_1, sigma_1)
    lliks_test2 = niw.log_likelihood(obs, mu_2, sigma_2)

    np.testing.assert_almost_equal(lliks_test1, lliks_true[:,0])
    np.testing.assert_almost_equal(lliks_test2, lliks_true[:,1])


def test_sufficient_statistics():
    N = 10
    mus_0 = np.array([[0.,0.], [20.,20.]])
    sigmas_0 = np.array([np.eye(2), np.eye(2)])
    kappa_0 = 0.5
    nu_0 = 5

    mu_1, sigma_1 = sample_niw(mus_0[0], sigmas_0[0], kappa_0, nu_0)
    mu_2, sigma_2 = sample_niw(mus_0[1], sigmas_0[1], kappa_0, nu_0)
    params = [[mu_1, sigma_1], [mu_2, sigma_2]]

    obs = np.array([mnorm.rvs(params[int(np.round(i/N))][0],
                              params[int(np.round(i/N))][1])
                    for i in xrange(1,N+1)]).astype(np.float64)

    rs1 = niw.expected_log_likelihood(obs, mus_0[0], sigmas_0[0], kappa_0, nu_0)
    rs2 = niw.expected_log_likelihood(obs, mus_0[1], sigmas_0[1], kappa_0, nu_0)
    rs = np.vstack((rs1, rs2)).T
    rs = np.exp(rs)
    rs /= np.sum(rs, axis=1)[:,np.newaxis]
    rs = rs.copy(order='C')

    s1s = np.sum(rs, axis=0)
    s2s = np.array([np.sum(obs*rs[:,i,np.newaxis], axis=0) for i in xrange(2)])
    s3s = np.array([np.sum([np.outer(obs[i],obs[i])*rs[i,0]
                            for i in xrange(obs.shape[0])],axis=0),
                    np.sum([np.outer(obs[i],obs[i])*rs[i,1]
                            for i in xrange(obs.shape[0])],axis=0)])

    s1, s2, s3 = niw.expected_sufficient_statistics(obs, rs[:,0].copy(order='C'))
    np.testing.assert_almost_equal(s1, s1s[0])
    np.testing.assert_almost_equal(s2, s2s[0])
    np.testing.assert_almost_equal(s3, s3s[0])

    s1, s2, s3 = niw.expected_sufficient_statistics(obs, rs[:,1].copy(order='C'))
    np.testing.assert_almost_equal(s1, s1s[1])
    np.testing.assert_almost_equal(s2, s2s[1])
    np.testing.assert_almost_equal(s3, s3s[1])


if __name__ == '__main__':
    #test_constructors()
    #test_update1()
    #test_update2()
    test_expected_log_likelihood()
    test_log_likelihood()
    test_sufficient_statistics()
    test_meanfield_sgd_update()
