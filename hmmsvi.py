from __future__ import division
from scipy.special import digamma

import numpy as np
import forward_backward as fb
import dirichlet as dir


class MetaObs(object):
    def __init__(self, i1, i2):
        self.i1 = i1
        self.i2 = i2


class HMMSVI(object):
    def __init__(self, obs, A_0, emits, tau, kappa):
        """

        Parameters
        ----------
        obs : np.ndarray
        A_0 : Distribution
        emits : list[Distribution]
        tau : float
        kappa : float
        """

        # TODO: might have to add epsilon for certain calculations involving the
        # transition matrix
        self.obs = obs
        self.A_nat_0 = A_0 - 1
        self.A_nat_N = self.A_nat_0.copy()
        self.emits = emits

        self.tau = tau
        self.kappa = kappa

        self.T = obs.shape[0]
        self.S = A_0.shape[0]

    def infer(self, L, n, metaobs_fn, itr):
        metaobs_sz = 2*L + 1

        for it in xrange(itr):
            lrate = (it + self.tau)**(-self.kappa)
            minibatches = metaobs_fn(self.T, L, n)

            A_inter = np.zeros_like(self.A_nat_N)
            emits_inter = [emit.zero_nat_params() for emit in self.emits]

            minibatches = metaobs_fn(self.T, L, metaobs_sz)

            for mb in minibatches:
                s_obs = self.obs[mb.i1:(mb.i2+1),:]
                pi = self._calc_pi()
                var_x = self.local_update(s_obs, pi)
                A_i, emits_i = self.intermediate_pars(s_obs, var_x)

                A_inter += A_i
                for i in xrange(len(emits_i)):
                    for j in xrange(len(emits_i[i])):
                        emits_inter[i][j] += emits_i[i][j]

            self.global_update(L, lrate, A_inter, emits_inter)

    def local_update(self, obs, pi):
        pi_mod = np.exp(digamma(pi) - digamma(np.sum(pi)))
        A_mod = np.exp(np.exp(dir.expected_sufficient_statistics(self.A_nat_N + 1)))
        elliks = np.array([emit.expected_log_likelihood(obs)
                           for emit in self.emits]).T.copy(order='C')

        lalpha = fb.forward_msgs(pi_mod, A_mod, elliks)
        lbeta  = fb.backward_msgs(A_mod, elliks)

        # TODO: encapsulate this more
        lvar_x, _ = fb.expected_statistics(pi, A_mod, elliks, lalpha, lbeta)
        var_x = np.exp(lvar_x)
        var_x /= np.sum(var_x, axis=1)[:,np.newaxis]

        return var_x

    def global_update(self, L, lrate, A_inter, emits_inter):
        # TODO: not sure about these batch factors
        S = 2*L + 1
        A_bfactor = (self.T-2*L-1)/(2*L*S)
        self.A_nat_N = dir.meanfield_sgd_update(self.A_nat_0, self.A_nat_N,
                                                A_inter, lrate, A_bfactor)

        e_bfactor = (self.T-2*L-1)/((2*L+1)*S)
        for i,emit in enumerate(self.emits):
            emit.meanfield_sgd_update(emits_inter[i], lrate, e_bfactor)

    def intermediate_pars(self, obs, var_x):
        A_inter  = dir.sufficient_statistics(var_x)
        A_inter -= 1

        emits_inter = [list(emit.expected_sufficient_statistics(obs,
                            var_x[:,i].copy(order='C')))
                       for i, emit in enumerate(self.emits)]

        return A_inter, emits_inter


    def _calc_pi(self):
        # This looks incorrect
        A_mean = (self.A_nat_N)/np.sum(self.A_nat_N, axis=1)
        ew, ev = np.linalg.eig(A_mean.T)
        ew_dec = np.argsort(ew)[::-1]
        return np.abs(ev[:,ew_dec[0]])

    @staticmethod
    def metaobs_unif(T, L, n):
        ll = L
        uu = T - 1 - L

        c_vec = np.random.randint(ll, uu+1, n)
        return [MetaObs(c-L,c+L) for c in c_vec]
