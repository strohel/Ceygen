# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Various support methods for tests"""

import numpy as np

import unittest as ut

class CeygenTestCase(ut.TestCase):
    """Test case that adds some numeric assert functions"""

    def assertApproxEqual(self, X, Y):
        """Return true if X = Y to within machine precision

        Function for checking that different matrices from different
        computations are in some sense "equal" in the verification tests.
        """
        X = np.asarray(X)
        Y = np.asarray(Y)
        fuzz = 1.0e-8

        self.assertEqual(X.ndim, Y.ndim)
        self.assertEqual(X.shape, Y.shape)

        if np.all(abs(X - Y) < fuzz):
            return
        else:
            self.fail("NumPy arrays {0} and {1} are not fuzzy equal (+- {2})".format(X, Y, fuzz))
