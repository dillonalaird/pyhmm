from __future__ import division

import abc
import numpy as np


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

    def __init__(self, obs):
        """
        Initialize the HMM object.

        Parameters
        ----------
        obs : numpy.array
            A T x D numpy array of T observations in D dimensions.
        """

        self.obs = obs

    def forward_msgs_debug(self, obs=None):
        """

        Parameters
        ----------
        obs : numpy.array
            A T x D numpy array of T observations in D dimensions.
        """

        ltran = self.mod_tran
        ll = self.lliks
        lalpha = self.lalpha
        lalpha[0,:] = self.mod_init + ll[0,:]

        for t in xrange(1, self.T):
            lalpha[t] = np.logaddexp.reduce(lalpha[t-1] + ltran.T, axis=1) + ll[t]

    def forward_msgs(self, obs=None):
        """

        Parameters
        ----------
        obs : numpy.array
            A T x D numpy array of T observations in D dimensions.
        """

        ltran = self.mod_tran
        ll = self.lliks
        lalpha = self.lalpha
        lalpha[0,:] = self.mod_init + ll[0,:]

    def backward_msgs(self, obs=None):
        """

        Parameters
        ----------
        obs : numpy.array
            A T x D numpy array of T observations in D dimensions.
        """

        pass
