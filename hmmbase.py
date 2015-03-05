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
        """

        self.obs = obs
        self.pi = pi
        self.A = A


    def baum_welch(self):
        # Expectation Step
        self.log_likelihoods()
        self.forward_msgs()
        self.backward_msgs()

        # Maximization Step

    def forward_msgs(self):
        """

        Parameters
        ----------
        obs : numpy.array
            A T x D numpy array of T observations in D dimensions.
        """

        self.lalpha = fb.forward_msgs(self.pi, self.A, self.lliks)

    def backward_msgs(self):
        """

        Parameters
        ----------
        obs : numpy.array
            A T x D numpy array of T observations in D dimensions.
        """

        self.lbeta = fb.backward_msgs(self.A, self.lliks)
