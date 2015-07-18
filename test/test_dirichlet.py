from scipy.special import digamma

import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
    os.pardir))

import numpy as np
import dirichlet as dir


def test_expected_sufficient_statistics():
    alphas = np.array([[1.,1.],
                       [1.,1.]])

    ess_true = digamma(alphas) - digamma(np.sum(alphas, axis=1)[:,np.newaxis])
    ess_pred = dir.expected_sufficient_statistics(alphas)
    np.testing.assert_almost_equal(ess_true, ess_pred)

    alphas = np.array([[1.,2.],
                       [3.,4.]])

    ess_true = digamma(alphas) - digamma(np.sum(alphas, axis=1)[:,np.newaxis])
    ess_pred = dir.expected_sufficient_statistics(alphas)
    np.testing.assert_almost_equal(ess_true, ess_pred)


if __name__ == '__main__':
    test_expected_sufficient_statistics()
