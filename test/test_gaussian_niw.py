from __future__ import division
import os, sys
from scipy.stats import multivariate_normal as mnorm
from scipy.special import digamma

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import numpy as np
import scipy.linalg
import scipy.stats as stats
import gaussian_niw as gniw

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

def _responsibilities(mu_0, sigma_0, kappa_0, nu_0, xs):
    D = mu_0.shape[0]
    x_bar = xs - mu_0
    log_lambda_tilde = _log_lambda_tilde(sigma_0, nu_0)
    xs = np.array([x_bar[i,:].T.dot(sigma_0).dot(x_bar[i,:])
                   for i in xrange(x_bar.shape[0])])
    return 0.5*(log_lambda_tilde - D*(1/kappa_0) - nu_0*xs - D*np.log(2*np.pi))

def _log_lambda_tilde(sigma_0, nu_0):
    D = sigma_0.shape[0]
    ln_sigma_0_det = np.log(np.linalg.det(sigma_0))
    return np.sum([digamma((nu_0 + 1 - i)/2) for i in xrange(D)]) - D*np.log(2) + \
        ln_sigma_0_det


def test_constructors():
    D = 3

    n1 = np.array([1,2,3]).astype(np.double)
    n2 = np.double(4)
    n3 = np.array([[5,6,7],[8,9,10],[11,12,13]]).astype(np.double)
    n4 = np.double(14)
    s1 = np.array([[1,1,1],[1,1,1],[1,1,1]]).astype(np.double)
    s2 = np.array([1,1,1]).astype(np.double)
    s3 = np.double(1)

    gniw.meanfield_update(n1, n2, n3, n4, s1, s2, s3)

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

    n1, n2, n3, n4 = gniw.meanfield_update(n1, n2, n3, n4, s1, s2, s3)

    print 'after n1 = ', n1
    print 'after n2 = ', n2
    print 'after n3 = ', n3
    print 'after n4 = ', n4


def test_update2():
    N = 100
    mus_0 = np.array([[0,0], [10,10]])
    sigmas_0 = np.array([np.eye(2), np.eye(2)])
    kappa_0 = 0.05
    nu_0 = 5

    mu_1, sigma_1 = _sample_niw(mus_0[0], sigmas_0[0], kappa_0, nu_0)
    mu_2, sigma_2 = _sample_niw(mus_0[1], sigmas_0[1], kappa_0, nu_0)
    #params = np.array([[mu_1, sigma_1], [mu_2, sigma_2]])
    params = np.array([mu_1, mu_2])

    obs = np.array([mnorm.rvs(params[np.round(i/N)], np.eye(2))
        for i in xrange(1,N+1)])

    rs1 = _responsibilities(mus_0[0], sigmas_0[0], kappa_0, nu_0, obs)
    rs2 = _responsibilities(mus_0[1], sigmas_0[1], kappa_0, nu_0, obs)
    rs = np.vstack((rs1, rs2)).T

    s1s = np.array([np.sum([np.outer(obs[i],obs[i])*rs[i,0]
                    for i in xrange(obs.shape[0])],axis=0),
                    np.sum([np.outer(obs[i],obs[i])*rs[i,1]
                    for i in xrange(obs.shape[0])],axis=0)])
    s2s = np.array([np.sum(obs*rs[:,i,np.newaxis], axis=0) for i in xrange(2)])
    s3s = np.sum(rs, axis=0)

    n1s = np.array([kappa_0*mus_0[0], kappa_0*mus_0[1]])
    n2s = np.array([kappa_0, kappa_0])
    n3s = np.array([sigmas_0[0] + kappa_0*np.outer(mus_0[0], mus_0[0]),
                    sigmas_0[1] + kappa_0*np.outer(mus_0[1], mus_0[1])])
    n4s = np.array([nu_0 + 2 + 2, nu_0 + 2 + 2])

    print 'label 1 before'
    print 'n1 = ', n1s[0]
    print 'n2 = ', n2s[0]
    print 'n3 = ', n3s[0]
    print 'n4 = ', n4s[0]

    n1, n2, n3, n4 = gniw.meanfield_update(n1s[0], n2s[0], n3s[0], n4s[0],
                                           s1s[0], s2s[0], s3s[0])

    print 'label 1 after'
    print 'n1 = ', n1
    print 'n2 = ', n2
    print 'n3 = ', n3
    print 'n4 = ', n4

    print 'label 2 before'
    print 'n1 = ', n1s[1]
    print 'n2 = ', n2s[1]
    print 'n3 = ', n3s[1]
    print 'n4 = ', n4s[1]

    n1, n2, n3, n4 = gniw.meanfield_update(n1s[1], n2s[1], n3s[1], n4s[1],
                                           s1s[1], s2s[1], s3s[1])

    print 'label 2 after'
    print 'n1 = ', n1
    print 'n2 = ', n2
    print 'n3 = ', n3
    print 'n4 = ', n4

if __name__ == '__main__':
    #test_constructors()
    test_update2()
