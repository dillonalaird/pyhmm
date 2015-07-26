from __future__ import division
from numpy.core.umath_tests import inner1d
from scipy.stats import multivariate_normal as mnorm
from scipy.special import digamma
from matplotlib import pyplot as plt

import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import numpy as np
import scipy.linalg
import scipy.stats as stats
import normal_invwishart as niw


def _sample_niw(mu, lmbda, kappa, nu):
    lmbda = _sample_invwishart(lmbda, nu)
    mu = np.random.multivariate_normal(mu, lmbda/kappa)
    return mu, lmbda


def _sample_nw(mu_0, sigma_0, kappa_0, nu_0):
    lmbda = _sample_wishart(sigma_0, nu_0)
    mu = np.random.multivariate_normal(mu_0, np.linalg.inv(lmbda*kappa_0))
    return mu, lmbda


def _sample_wishart(sigma_0, nu_0):
    # if A ~ W(sigma_0, nu_0) then X = A^-1 X ~ W^-1(simga_0^-1, nu_0)
    return _sample_invwishart(np.linalg.inv(sigma_0), nu_0)


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


def _responsibilities(mu_0, sigma_0, kappa_0, nu_0, xs):
    # this is responsibilities for niw so we invert sigma_0
    D = mu_0.shape[0]
    sigma_0 = np.linalg.inv(sigma_0)
    x_bar = xs - mu_0
    log_lambda_tilde = _log_lambda_tilde(sigma_0, nu_0)
    xs = np.array([x_bar[i,:].T.dot(sigma_0).dot(x_bar[i,:])
                   for i in xrange(x_bar.shape[0])])
    return 0.5*(log_lambda_tilde - D*(1/kappa_0) - nu_0*xs - D*np.log(2*np.pi))


def _log_lambda_tilde(sigma_0, nu_0):
    D = sigma_0.shape[0]
    ln_sigma_0_det = np.log(np.linalg.det(sigma_0))
    return np.sum([digamma((nu_0 + 1 - i)/2) for i in xrange(D)]) + D*np.log(2) + \
        ln_sigma_0_det


def _responsibilities2(mu_0, sigma_0, kappa_0, nu_0, xs):
    D = mu_0.shape[0]
    chol = np.linalg.cholesky(sigma_0)
    xs = np.reshape(xs,(-1,D))
    xs = np.linalg.solve(chol, xs.T)
    return 0.5*(_log_lambda_tilde2(sigma_0, nu_0) - D*(1/kappa_0) - nu_0*
                inner1d(xs.T,xs.T) - D*np.log(2*np.pi))


def _log_lambda_tilde2(sigma_0, nu_0):
    D = sigma_0.shape[0]
    chol = np.linalg.cholesky(sigma_0)
    return digamma((nu_0 - np.arange(D))/2.).sum() \
        + D*np.log(2) + np.log(chol.diagonal()).sum()


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


def test_constructors():
    D = 3

    n1 = np.array([1,2,3]).astype(np.double)
    n2 = np.double(4)
    n3 = np.array([[5,6,7],[8,9,10],[11,12,13]]).astype(np.double)
    n4 = np.double(14)
    s1 = np.array([[1,1,1],[1,1,1],[1,1,1]]).astype(np.double)
    s2 = np.array([1,1,1]).astype(np.double)
    s3 = np.double(1)

    niw.meanfield_update(n1, n2, n3, n4, s1, s2, s3)

    print 'after n1 = ', n1
    print 'after n2 = ', n2
    print 'after n3 = ', n3
    print 'after n4 = ', n4


def test_update1():
    D = 3

    n1 = np.array([1,2,3]).astype(np.double)
    n2 = np.double(4)
    n3 = np.array([[5,6,7],[8,9,10],[11,12,13]]).astype(np.double)
    n4 = np.double(14)
    s1 = np.array([[1,1,1],[1,1,1],[1,1,1]]).astype(np.double)
    s2 = np.array([1,1,1]).astype(np.double)
    s3 = np.double(1)

    n1, n2, n3, n4 = niw.meanfield_update(n1, n2, n3, n4, s1, s2, s3)

    print 'after n1 = ', n1
    print 'after n2 = ', n2
    print 'after n3 = ', n3
    print 'after n4 = ', n4


def test_update2():
    N = 100
    mus_0 = np.array([[0,0], [20,20]])
    sigmas_0 = np.array([np.eye(2), np.eye(2)])
    kappa_0 = 0.5
    nu_0 = 5

    mu_1, sigma_1 = _sample_niw(mus_0[0], sigmas_0[0], kappa_0, nu_0)
    mu_2, sigma_2 = _sample_niw(mus_0[1], sigmas_0[1], kappa_0, nu_0)
    params = [[mu_1, sigma_1], [mu_2, sigma_2]]

    print 'sampled params'
    print '\tmu_1 = ', mu_1
    print '\tsigma_1 = ', (1/kappa_0)*sigma_1
    print '\tmu_2 = ', mu_2
    print '\tsigma_2 = ', (1/kappa_0)*sigma_2

    obs = np.array([mnorm.rvs(params[int(np.round(i/N))][0],
                              params[int(np.round(i/N))][1])
                    for i in xrange(1,N+1)])

    plt.scatter(obs[:,0], obs[:,1])
    plt.show()

    rs1 = _responsibilities(mus_0[0], sigmas_0[0], kappa_0, nu_0, obs)
    rs2 = _responsibilities(mus_0[1], sigmas_0[1], kappa_0, nu_0, obs)
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

    print 'label 1 before'
    print '\tmu_0 = ', mus_0[0]
    print '\tsigma_0 = ', sigmas_0[0]
    print '\tkappa_0 = ', kappa_0
    print '\tnu_0 = ', nu_0

    n1, n2, n3, n4 = niw.meanfield_update(n1s[0], n2s[0], n3s[0], n4s[0],
                                          s1s[0], s2s[0], s3s[0])

    mu_0, sigma_0, kappa_0, nu_0 = natural_to_standard(n1, n2, n3, n4)

    mu_1, sigma_1 = _sample_niw(mu_0, sigma_0, kappa_0, nu_0)

    print 'label 1 after'
    print '\tmu_0 = ', mu_0
    print '\tsigma_0 = ', sigma_0
    print '\tkappa_0 = ', kappa_0
    print '\tnu_0 = ', nu_0

    print 'label 2 before'
    print '\tmu_0 = ', mus_0[1]
    print '\tsigma_0 = ', sigmas_0[1]
    print '\tkappa_0 = ', kappa_0
    print '\tnu_0 = ', nu_0

    n1, n2, n3, n4 = niw.meanfield_update(n1s[1], n2s[1], n3s[1], n4s[1],
                                          s1s[1], s2s[1], s3s[1])

    mu_0, sigma_0, kappa_0, nu_0 = natural_to_standard(n1, n2, n3, n4)

    print 'label 2 after'
    print '\tmu_0 = ', mu_0
    print '\tsigma_0 = ', sigma_0
    print '\tkappa_0 = ', kappa_0
    print '\tnu_0 = ', nu_0

    mu_2, sigma_2 = _sample_niw(mu_0, sigma_0, kappa_0, nu_0)

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

    mu_1, sigma_1 = _sample_niw(mus_0[0], sigmas_0[0], kappa_0, nu_0)
    mu_2, sigma_2 = _sample_niw(mus_0[1], sigmas_0[1], kappa_0, nu_0)
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

    rs1_true = _responsibilities(mus_0[0], sigmas_0[0], kappa_0, nu_0, obs)
    rs2_true = _responsibilities(mus_0[1], sigmas_0[1], kappa_0, nu_0, obs)

    np.testing.assert_almost_equal(rs1, rs1_true)
    np.testing.assert_almost_equal(rs2, rs2_true)


def test_log_likelihood():
    N = 10
    mus_0 = np.array([[0.,0.], [20.,20.]])
    sigmas_0 = np.array([np.eye(2), np.eye(2)])
    kappa_0 = 0.5
    nu_0 = 5

    mu_1, sigma_1 = _sample_niw(mus_0[0], sigmas_0[0], kappa_0, nu_0)
    mu_2, sigma_2 = _sample_niw(mus_0[1], sigmas_0[1], kappa_0, nu_0)
    params = [[mu_1, sigma_1], [mu_2, sigma_2]]

    obs = np.array([mnorm.rvs(params[int(np.round(i/N))][0],
                              params[int(np.round(i/N))][1])
                    for i in xrange(1,N+1)]).astype(np.float64)

    lliks_true = np.array([[mnorm.logpdf(ob, mu_1, sigma_1),
                            mnorm.logpdf(ob, mu_2, sigma_2)] for ob in obs])
    lliks_test1 = niw.log_likelihood(obs, mu_1, sigma_1)
    lliks_test2 = niw.log_likelihood(obs, mu_2, sigma_2)

    np.testing.assert_almost_equal(lliks_true[:,0], lliks_test1)
    np.testing.assert_almost_equal(lliks_true[:,1], lliks_test2)


def test_sufficient_statistics():
    N = 10
    mus_0 = np.array([[0.,0.], [20.,20.]])
    sigmas_0 = np.array([np.eye(2), np.eye(2)])
    kappa_0 = 0.5
    nu_0 = 5

    mu_1, sigma_1 = _sample_niw(mus_0[0], sigmas_0[0], kappa_0, nu_0)
    mu_2, sigma_2 = _sample_niw(mus_0[1], sigmas_0[1], kappa_0, nu_0)
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
    np.testing.assert_almost_equal(s1s[0], s1)
    np.testing.assert_almost_equal(s2s[0], s2)
    np.testing.assert_almost_equal(s3s[0], s3)

    s1, s2, s3 = niw.expected_sufficient_statistics(obs, rs[:,1].copy(order='C'))
    np.testing.assert_almost_equal(s1s[1], s1)
    np.testing.assert_almost_equal(s2s[1], s2)
    np.testing.assert_almost_equal(s3s[1], s3)


if __name__ == '__main__':
    #test_constructors()
    #test_update2()
    #test_expected_log_likelihood()
    #test_log_likelihood()
    #test_sufficient_statistics()
    test_meanfield_sgd_update()
