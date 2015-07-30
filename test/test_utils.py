from __future__ import division
from numpy.core.umath_tests import inner1d
from scipy.special import digamma
from scipy.stats import multivariate_normal as mnorm

import numpy as np
import scipy.linalg
import scipy.stats as stats


def sample_niw(mu, lmbda, kappa, nu):
    lmbda = sample_invwishart(lmbda, nu)
    mu = np.random.multivariate_normal(mu, lmbda/kappa)
    return mu, lmbda


def sample_nw(mu_0, sigma_0, kappa_0, nu_0):
    lmbda = sample_wishart(sigma_0, nu_0)
    mu = np.random.multivariate_normal(mu_0, np.linalg.inv(lmbda*kappa_0))
    return mu, lmbda


def sample_wishart(sigma_0, nu_0):
    # if A ~ W(sigma_0, nu_0) then X = A^-1 X ~ W^-1(simga_0^-1, nu_0)
    return sample_invwishart(np.linalg.inv(sigma_0), nu_0)


def sample_invwishart(S, nu):
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


def responsibilities(mu_0, sigma_0, kappa_0, nu_0, xs):
    # this is responsibilities for niw so we invert sigma_0
    D = mu_0.shape[0]
    sigma_0 = np.linalg.inv(sigma_0)
    x_bar = xs - mu_0
    log_lmbda_tilde = log_lambda_tilde(sigma_0, nu_0)
    xs = np.array([x_bar[i,:].T.dot(sigma_0).dot(x_bar[i,:])
                   for i in xrange(x_bar.shape[0])])
    return 0.5*(log_lmbda_tilde - D*(1/kappa_0) - nu_0*xs - D*np.log(2*np.pi))


def log_lambda_tilde(sigma_0, nu_0):
    D = sigma_0.shape[0]
    ln_sigma_0_det = np.log(np.linalg.det(sigma_0))
    return np.sum([digamma((nu_0 + 1 - i)/2) for i in xrange(1,D+1)]) + D*np.log(2) + \
        ln_sigma_0_det


def _compare(mu_0, sigma_0, kappa_0, nu_0, xs):
    D = mu_0.shape[0]
    chol = np.linalg.cholesky(sigma_0)
    xs1 = np.reshape(xs,(-1,D)) - mu_0
    xs1 = np.linalg.solve(chol, xs1.T)
    final1 = inner1d(xs1.T, xs1.T)

    sigma_inv = np.linalg.inv(sigma_0)
    x_bar = xs - mu_0
    final2 = np.array([x_bar[i,:].T.dot(sigma_inv).dot(x_bar[i,:])
                      for i in xrange(x_bar.shape[0])])

    print final1
    print final2
    print ''
    print log_lambda_tilde2(sigma_0, nu_0)
    print log_lambda_tilde(sigma_inv, nu_0)

    print np.log(chol.diagonal()).sum()
    print np.log(np.linalg.det(sigma_inv))

    print np.sum([digamma((nu_0 + 1 - i)/2.) for i in xrange(1,D+1)])
    print digamma((nu_0 - np.arange(D))/2.).sum()


def responsibilities2(mu_0, sigma_0, kappa_0, nu_0, xs):
    D = mu_0.shape[0]
    chol = np.linalg.cholesky(sigma_0)
    xs = np.reshape(xs,(-1,D)) - mu_0
    xs = np.linalg.solve(chol, xs.T)
    return 0.5*(log_lambda_tilde2(sigma_0, nu_0) - D*(1/kappa_0) - nu_0*
                inner1d(xs.T,xs.T) - D*np.log(2*np.pi))


def log_lambda_tilde2(sigma_0, nu_0):
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


def generate_data(D, N, pi, A, params):
    s = np.random.choice(D, 1, p=pi)[0]
    obs = [mnorm.rvs(params[s][0], params[s][1])]
    for i in xrange(N-1):
        s = np.random.choice(2, 1, p=A[s,:])[0]
        obs.append(mnorm.rvs(params[s][0], params[s][1]))

    return np.array(obs)
