from __future__ import division

import numpy as np
import forward_backward as fb
import dirichlet as dir


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

        self.obs = obs
        self.A_nat_0 = A_0 - 1
        self.A_nat_N = self.A_nat_0.copy()
        self.emits = emits

        self.tau = tau
        self.kappa = kappa

        self.T = obs.shape[0]
        self.S = A_0.size()

    def infer(self, L, n, metaobs_fn, itr):
        metaobs_sz = 2*L + 1

        for it in xrange(itr):
            lrate = (it + self.tau)**(-self.kappa)
            minibatches = metaobs_fn(self.T, L, n)

            A_inter = np.zeros_like(self.A_nat_N)
            emits_inter = [emit.zero_nat_params() for emit in self.emits]

            for mb in minibatches:
                s_obs = self.obs[mb.i1:(mb.i2+1)]
                pi = self._calc_pi()
                var_x = self.local_update(s_obs, pi)
                A_i, emits_i = self.intermediate_pars(s_obs, var_x)

                A_inter += A_i
                for s in xrange(emits_i.shape[0]):
                    emits_inter[s] += emits_i[s]

            self.global_update(L, lrate, A_inter, emits_inter)

    def local_update(self, obs, pi):
        A_mod = np.exp(dir.expected_sufficient_statistics(self.A_nat_N + 1))
        elliks = np.array([emit.expected_log_likelihood(obs)
                           for emit in self.emits])

        lalpha = fb.forward_msgs(pi, A_mod, elliks)
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

        emits_inter = np.array([list(emit.expected_sufficient_statistics(obs, var_x[:,i]))
                                for i, emit in enumerate(self.emits)])

        return A_inter, emits_inter


    def _calc_pi(self):
        A_mean = (self.A_nat_N + 1)/np.sum(self.A_nat_N + 1, axis=1)
        ew, ev = np.linalg.eigen(A_mean.T)
        ew_dec = np.argsort(ew)[::-1]
        return np.abs(ev[:,ew_dec[0]])
