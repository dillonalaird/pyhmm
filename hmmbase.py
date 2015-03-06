from __future__ import division

import abc
import numpy as np
import forward_backward as fb


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

    def __init__(self, obs, K, pi, A):
        """
        Initialize the HMM object.

        Parameters
        ----------
        obs : numpy.array
            A T x D numpy array of T observations in D dimensions.

        K : int
            The number of hidden states to use.

        pi : numpy.array
            A 1 x K array representing the initial distribution.

        A : numpy.array
            A K x K matrix representing the transition matrix.
        """

        self.obs = obs
        self.pi = pi
        self.A = A


    def EM(self):
        # Expectation Step
        self.log_likelihoods()
        self.forward_msgs()
        self.backward_msgs()
        self.expected_statistics()

        # Maximization Step

    def forward_msgs(self):
        """
        """

        self.lalpha = fb.forward_msgs(self.pi, self.A, self.lliks)

    def backward_msgs(self):
        """
        """

        self.lbeta = fb.backward_msgs(self.A, self.lliks)

    def expected_statistics(self):
        """
        """

        self.expected_states, self.expected_transcounts = \
            fb.expected_statistics(self.pi, self.A, self.lliks, self.lalpha, 
                                   self.lbeta)
