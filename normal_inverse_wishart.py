import numpy as np
import normal_invwishart as niw


class NormalInvWishart(object):
    def __init__(self, mu_0, sigma_0, kappa_0, nu_0):
        self.mu_0    = self.mu_N    = mu_0
        self.sigma_0 = self.sigma_N = sigma_0
        self.kappa_0 = self.kappa_N = kappa_0
        self.nu_0    = self.nu_N    = nu_0
        self.D = mu_0.shape[0]

    def expected_sufficient_statistics(self, obs, expected_states):
        return niw.expected_sufficient_statistics(obs, expected_states)

    # TODO: clean this up, move to normal_invwishart_interface.pyx
    def meanfield_sgd_update(self, ess, lrate, bfactor):
        nat_params_0 = self.shape_nat_params(*self.standard_to_natural(
            self.mu_0, self.sigma_0, self.kappa_0, self.nu_0))
        nat_params_N = self.shape_nat_params(*self.standard_to_natural(
            self.mu_N, self.sigma_N, self.kappa_N, self.nu_N))
        new_nats = niw.meanfield_sgd_update(nat_params_0, nat_params_N, ess[0],
                                            ess[1], ess[2], lrate, bfactor)
        self.mu_N, self.sigma_N, self.kappa_N, self.nu_N = \
            self.natural_to_standard(*self.unshape_nat_params(new_nats))

    def zero_nat_params(self):
        D = self.D
        return np.array([np.zeros(D), 0., np.zeros((D,D)), 0.])

    def standard_to_natural(self, mu, sigma, kappa, nu):
        n1 = kappa*mu
        n2 = kappa
        n3 = sigma + kappa*np.outer(mu, mu)
        n4 = nu + 2 + mu.shape[0]
        return n1, n2, n3, n4

    def natural_to_standard(self, n1, n2, n3, n4):
        kappa = n2
        mu    = n1/n2
        sigma = n3 - kappa*np.outer(mu, mu)
        nu    = n4 - 2 - mu.shape[0]
        return mu, sigma, kappa, nu

    # TODO: this should be done in normal_invwishart_interface.pyx
    def shape_nat_params(self, n1, n2, n3, n4):
        D = n1.shape[0]
        nat_params = np.zeros((D+3,D))
        nat_params[:D,:D] = n3
        nat_params[D,:]   = n1
        nat_params[D+1,0] = n2
        nat_params[D+2,0] = n4
        return nat_params

    def unshape_nat_params(self, nat_params):
        D = nat_params.shape[0] - 3
        n3 = nat_params[:D,:D]
        n1 = nat_params[D,:]
        n2 = nat_params[D+1,0]
        n4 = nat_params[D+2,0]
        return n1, n2, n3, n4
