from __future__ import division
import os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import numpy as np
import gaussian_niw as gniw


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


def test_update():
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

if __name__ == '__main__':
    #test_constructors()
    test_update()
