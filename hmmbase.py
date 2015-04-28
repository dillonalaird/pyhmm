from __future__ import division

import abc
import numpy as np
import forward_backward as fb
import gmm


class HMMBase(object):
    """
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def infer():
        """
        Perform inference.
        """

        pass

    def __init__(self, obs, S, pi, A, L):
        """
        Initialize the HMM object.

        Parameters
        ----------
        obs : numpy.array
            A T x D numpy array of T observations in D dimensions.

        S : int
            The number of hidden states to use.

        pi : numpy.array
            A 1 x K array representing the initial distribution.

        A : numpy.array
            A K x K matrix representing the transition matrix.

        L : int
            The number of mixtures for the Gaussian Mixture Model.
        """

        self.obs = obs
        self.pi = pi
        self.A = A

        # initialize GMM parameters
        self.cs
        self.mus
        self.sigmas

    def Baum_Welch(self):
        for itr in iters:
            # Expectation Step
            self._forward_msgs()
            self._backward_msgs()
            self._log_weights()
            lexpected_states, expected_transcounts = self._expected_statistics()

            # Maximization Step
            expected_states = np.exp(lexpected_states)
            expected_states = epxected_states/np.sum(expected_states, axis=1)[:,np.newaxis]

            self.pi = expected_states[0]
            self.A = expected_transcounts/ \
                    np.sum(expected_states[:len(expected_states)-1], axis=0)[:,np.newaxis]

            gmm.update_parameters(self.cs, self.mus, self.sigmas, self.obs, self.lweights)


    def _forward_msgs(self):
        """
        """

        self.lalpha = fb.forward_msgs(self.pi, self.A, self.lliks)

    def _backward_msgs(self):
        """
        """

        self.lbeta = fb.backward_msgs(self.A, self.lliks)

    def _log_likelihood(self):
        """
        """

        pass

    def _log_weights(self):
        """
        """

        self.lweights = fb.log_weights(self.lalpha, self.lbeta)

    def _expected_statistics(self):
        """
        """

        return fb.expected_statistics(self.pi, self.A, self.lliks,
                self.lalpha, self.lbeta)
