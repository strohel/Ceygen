# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

import numpy as np

from support import CeygenTestCase, skipIfEigenOlderThan
cimport ceygen.llt as llt


class TestLlt(CeygenTestCase):

    def test_cholesky(self):
        x_np = np.array([[1., 0., 0.], [0., 4., 0.], [0., 0., 25.]])
        expected = np.array([[1., 0., 0.], [0., 2., 0.], [0., 0., 5.]])

        self.assertApproxEqual(llt.cholesky[double](x_np), expected)
        out_np = np.empty((3, 3))
        out2 = llt.cholesky[double](x_np, out_np)
        self.assertApproxEqual(out_np, expected)  # test that it actually uses out
        self.assertApproxEqual(out2, expected)

        cdef double[:, :] x = x_np
        self.assertApproxEqual(llt.cholesky(x), expected)
        cdef double[:, :] out = np.empty((3, 3))
        out2 = llt.cholesky(x, out)
        self.assertApproxEqual(out, expected)
        self.assertApproxEqual(out2, expected)

    @skipIfEigenOlderThan(3, 1, 0)
    def test_cholesky_badinput(self):
        x = 0.5 * np.eye(2)
        out = np.zeros((2, 2))
        for X in (x, None, np.array([1.]), np.array([[1.], [2.]])):
            for OUT in (out, None, np.empty(2), np.empty((3, 2))):
                if X is x and (OUT is out or OUT is None):
                    llt.cholesky[double](X, OUT)  # this should be valid
                else:
                    with self.assertRaises(ValueError):
                        llt.cholesky[double](X, OUT)  # this should be valid
